"""Bảng cuối 50 prompt: chính (A · I0 · SAVIER) + ablation (I0 · refs-only · keep-only · SAVIER) với VQA/VCFS/CCR/NRG + CAIRE-VN/CN.
SAVIER = lô chính sau no-op fix (metrics_final2_*). CAIRE: A/I0 từ caire_300.json (lượt 8-culture), I1/refs/keep từ caire_abl.json (2-culture; điểm từng culture độc lập).

    python scripts/final_table.py --dir docs/report_assets/metrics_final --caire-old ../SAVIER/results/metrics_50/caire_300.json --caire-new docs/report_assets/caire_abl/caire_abl.json -o results/metrics/final_50.md
"""
import argparse, json, random, statistics as st
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--caire-old", required=True); ap.add_argument("--caire-new", required=True); ap.add_argument("-o", required=True); a = ap.parse_args()
D = Path(a.dir); rng = random.Random(0); CO = json.loads(Path(a.caire_old).read_text()); CN = json.loads(Path(a.caire_new).read_text())
def load(n): return {u["pid"]: u for u in json.loads((D / n / "metrics.json").read_text(encoding="utf-8"))["don_vi"]}
def ci(xs, B=2000):
    xs = [x for x in xs if x is not None]; m = st.mean(xs); bs = sorted(st.mean(rng.choice(xs) for _ in xs) for _ in range(B)); return m, bs[int(.025 * B)], bs[int(.975 * B) - 1]
def caire(lab_old, lab_new, pid, arm):
    if arm in ("A", "I0"): v = CO.get(f"{lab_old}|{pid}|{arm}"); return (v[0], v[1]) if v else (None, None)
    v = CN.get(f"{lab_new}|{pid}|I1"); return (v["Vietnam"], v["China"]) if v else (None, None)
def nrg(us): v = [u["NRG"] for u in us.values() if u.get("NRG") is not None]; return f"{sum(x > 0 for x in v)}/{sum(x == 0 for x in v)}/{sum(x < 0 for x in v)}"
out = ["# Kết quả cuối — 50 prompt, một seed (SAVIER = arm chính sau no-op fix; evaluator Qwen2.5-VL-7B mù; CAIRE mã gốc, Qwen2.5-VL scorer)", ""]
for model, lab_old, main, ro, ko in (("SDXL 1.0", "SDXL", "metrics_final2_sdxl", "metrics_abl_sdxl_refsonly", "metrics_abl_sdxl_keeponly"), ("FLUX.1-dev", "FLUX.1-dev", "metrics_final2_flux", "metrics_abl_flux_refsonly", "metrics_abl_flux_keeponly")):
    M, R, K = load(main), load(ro), load(ko); pids = sorted(M); ln = "SDXL" if lab_old == "SDXL" else "FLUX"
    rows = [("A · prompt gốc", M, "A", lab_old, None), ("I0 · refined prompt", M, "I0", lab_old, None), ("+ reference (refs-only)", R, "I1", None, ln + "-refs"), ("+ keep-only", K, "I1", None, ln + "-keep"), ("**SAVIER (ours)**", M, "I1", None, ln)]
    out += [f"## {model}", "", "| arm | CAIRE-VN ↑ | CAIRE-CN ↓ | VQAScore ↑ | VCFS ↑ | CCR ↓ | NRG >0/=0/<0 |", "|---|---:|---:|---:|---:|---:|---|"]
    vals = {}
    for name, us, arm, lo, lnew in rows:
        cv = [caire(lo, lnew, p, arm) for p in pids]; vn, cn = ci([x[0] for x in cv]), ci([x[1] for x in cv])
        f = lambda k: ci([us[p]["diem"][arm][k] for p in pids]); vq, vc, cc = f("VQA"), f("VCFS"), f("CCR"); vals[name] = (cv, us, arm)
        out.append(f"| {name} | {vn[0]:.2f} | {cn[0]:.2f} | {vq[0]:.3f} [{vq[1]:.2f}, {vq[2]:.2f}] | {vc[0]:.1f} [{vc[1]:.1f}, {vc[2]:.1f}] | {cc[0]:.1f} [{cc[1]:.1f}, {cc[2]:.1f}] | {nrg(us) if arm == 'I1' else '–'} |")
    out += ["", "Hiệu số ghép đôi theo prompt (SAVIER − arm), 95% CI bootstrap:", "", "| so với | ΔCAIRE-VN | ΔCAIRE-CN | ΔVQA | ΔVCFS | ΔCCR |", "|---|---:|---:|---:|---:|---:|"]
    S = vals["**SAVIER (ours)**"]
    for name in ("I0 · refined prompt", "+ reference (refs-only)", "+ keep-only"):
        T = vals[name]; d = lambda i: ci([S[0][j][i] - T[0][j][i] for j in range(len(pids)) if S[0][j][i] is not None and T[0][j][i] is not None])
        g = lambda k: ci([S[1][p]["diem"]["I1"][k] - T[1][p]["diem"][T[2]][k] for p in pids])
        dv, dc, q, v, c = d(0), d(1), g("VQA"), g("VCFS"), g("CCR")
        out.append(f"| {name} | {dv[0]:+.2f} [{dv[1]:+.2f}, {dv[2]:+.2f}] | {dc[0]:+.2f} [{dc[1]:+.2f}, {dc[2]:+.2f}] | {q[0]:+.3f} [{q[1]:+.2f}, {q[2]:+.2f}] | {v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}] | {c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}] |")
    out.append("")
Path(a.o).parent.mkdir(parents=True, exist_ok=True); Path(a.o).write_text("\n".join(out), encoding="utf-8"); print("\n".join(out))
