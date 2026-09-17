import sys, os, glob, json
sys.path.insert(0, "/workspace/ctig17")
exec(open("/workspace/score_rules.py").read().split("print(\"%-12s")[0])

def drop_rules(rm, rspread):
    return {"chi mean<0.5": [a for a in HAVE if rm[a] < 0.50],
            "mean<0.5 hoac spread>0.45": [a for a in HAVE if rm[a] < 0.50 or rspread[a] > 0.45],
            "chi spread>0.35": [a for a in HAVE if rspread[a] > 0.35]}

print("%-11s %-28s %6s %-16s %-14s %s" % ("anh", "tieu chi loai", "AUC", "DUNG thap nhat", "SAI cao nhat", "bo"))
for mode in ("đầy đủ", "cắt người"):
    d = data[mode]
    rm = {a: sum(d["ref"][a])/len(d["ref"][a]) for a in ATTRS}
    rsp = {a: max(d["ref"][a]) - min(d["ref"][a]) for a in ATTRS}
    for lab, dead in drop_rules(rm, rsp).items():
        live = [a for a in HAVE if a not in dead]
        if not live: live = HAVE
        w = lambda a: 2.0 if a in HAVE[:2] else 1.0
        sc = {}
        for n, vals in d["img"].items():
            tot = sum(w(a) for a in live) or 1.0
            s = sum(w(a) for a in live if vals[a] >= max(0.55, rm[a] - 0.08))
            pen = sum(1.0 for a in NOT if vals[a] >= max(0.70, rm[a] + 0.20))
            sc[n] = (s - pen)/tot
        a_, lo, hi = auc(sc)
        print("%-11s %-28s %6.2f %-16.2f %-14.2f %s" % (mode, lab, a_, lo, hi, [x[:22] for x in dead]))
print()
for mode in ("đầy đủ", "cắt người"):
    d = data[mode]
    print(mode, "gia tri tren 3 anh that:")
    for a in HAVE:
        print("   %-52s %s  spread %.2f" % (a[:52], ["%.2f" % x for x in d["ref"][a]], max(d["ref"][a])-min(d["ref"][a])))
