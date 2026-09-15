"""
Rank agent (v1.4): xếp top-k ứng viên còn lại sau Filter, rồi đối chiếu với xếp hạng theo metric.

"Debate" ở mức cơ bản = hai bên xếp độc lập (agent từ mô tả + brief; metric từ CLIP attr/ITM attr/PickScore),
báo cáo điểm đồng thuận (top-1 trùng không, Spearman), thứ tự cuối = trung bình hạng, hoà thì theo metric.
"""

from __future__ import annotations

from ..schema import Candidate, CulturalBrief, CulturalSpec, FilterResult, RankResult


def _spearman(a: list[str], b: list[str]) -> float | None:
    common = [x for x in a if x in b]
    n = len(common)
    if n < 3:
        return None
    ra = {x: i for i, x in enumerate([x for x in a if x in common])}
    rb = {x: i for i, x in enumerate([x for x in b if x in common])}
    d2 = sum((ra[x] - rb[x]) ** 2 for x in common)
    return round(1 - 6 * d2 / (n * (n * n - 1)), 3)


def rm_idx(order: list[str], p: str) -> int:
    return order.index(p) if p in order else 99


def run(agent, cands: list[tuple[Candidate, str]], flt: FilterResult, briefs: dict[str, CulturalBrief],
        spec: CulturalSpec, prompt_en: str, log=print) -> RankResult:
    """cands: [(Candidate, model_key)] đã xếp theo metric giảm dần (combined_score)."""
    from ..stages.multigen import combined_score

    kept = set(flt.kept)
    pool = [(c, m) for c, m in cands if c.path in kept]
    order_metric = [c.path for c, _ in sorted(pool, key=lambda cm: -combined_score(cm[0]))]
    desc_by = {d.path: d for d in flt.descriptors}
    ver_by = {v.path: v for v in flt.verdicts}
    items = []
    for i, (c, m) in enumerate(pool):
        d, v = desc_by.get(c.path), ver_by.get(c.path)
        items.append({"id": f"img{i + 1}", "model": m, "description": d.text() if d else "",
                      "must_have_seen": v.matched_must_have if v else [], "must_not_seen": v.matched_must_not if v else [],
                      "metric_score": round(combined_score(c), 3)})
    id2path = {it["id"]: c.path for it, (c, _) in zip(items, pool)}
    brief_txt = "\n".join(f"- {se.name_en}: " + "; ".join((briefs.get(se.entity_id).facts_en if briefs.get(se.entity_id) else se.required_attrs_en)[:4])
                          for se in spec.entities)
    order_agent, reasons = list(order_metric), {}
    if len(pool) >= 2:
        try:
            # Hai lượt, lượt 2 đảo thứ tự trình bày (position swap - "VLM judges can rank but cannot score"):
            # hạng cuối = trung bình hạng hai lượt, khử thiên lệch vị trí của LLM.
            orders = []
            for items_view in (items, list(reversed(items))):
                d = agent.rank_candidates(prompt_en, brief_txt, items_view)
                got = [id2path[i] for i in d.get("order", []) if i in id2path]
                got += [p for p in order_metric if p not in got]  # thiếu id nào thì nối theo metric
                orders.append(got)
                if not reasons:
                    reasons = {id2path[k]: str(v)[:200] for k, v in (d.get("reasons") or {}).items() if k in id2path}
            avg = {p: sum(o.index(p) for o in orders) / len(orders) for p in order_metric}
            order_agent = sorted(order_metric, key=lambda p: (avg[p], rm_idx(order_metric, p)))
        except Exception as exc:  # noqa: BLE001
            log(f"  [rank] agent lỗi ({type(exc).__name__}: {str(exc)[:80]}) -> dùng thứ tự metric")
    # trung bình hạng
    ra = {p: i for i, p in enumerate(order_agent)}
    rm = {p: i for i, p in enumerate(order_metric)}
    final = sorted(order_metric, key=lambda p: (ra.get(p, 99) + rm[p], rm[p]))
    dis = []
    if order_agent and order_metric and order_agent[0] != order_metric[0]:
        dis.append(f"agent chọn {_short(order_agent[0])}, metric chọn {_short(order_metric[0])}")
    for p in order_metric[:3]:
        if abs(ra.get(p, 99) - rm[p]) >= 3:
            dis.append(f"{_short(p)}: hạng metric {rm[p] + 1}, hạng agent {ra.get(p, 99) + 1}")
    rr = RankResult(order_agent=order_agent, order_metric=order_metric, final_order=final, reasons=reasons,
                    agreement_top1=bool(order_agent and order_metric and order_agent[0] == order_metric[0]),
                    spearman=_spearman(order_agent, order_metric), disagreements=dis[:4])
    log(f"  [rank] {len(pool)} ứng viên, top-1 {'trùng' if rr.agreement_top1 else 'KHÁC'}, Spearman {rr.spearman}")
    return rr


def _short(p: str) -> str:
    from pathlib import Path

    return Path(p).stem
