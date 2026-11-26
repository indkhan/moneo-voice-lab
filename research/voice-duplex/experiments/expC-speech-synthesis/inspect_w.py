import sys, json, numpy as np
W = sys.argv[1]
z = np.load(W)
tree = json.loads(str(z["__tree__"]))
def show(n, d=0, name=""):
    if isinstance(n, dict):
        if "@" in n and len(n)==1:
            a = z[n["@"]]; print("  "*d + f"{name}: arr {a.shape} {a.dtype}"); return
        print("  "*d + f"{name}: dict")
        for k,v in n.items(): show(v, d+1, k)
    elif isinstance(n, list):
        if len(n) > 6 and not isinstance(n[0], (dict, list)):
            print("  "*d + f"{name}: list len {len(n)} {n[:6]}"); return
        print("  "*d + f"{name}: list[{len(n)}]")
        for i,v in enumerate(n[:40]):
            show(v, d+1, str(i))
            if d>=2 and i>=1 and name in ("layers","tf","hblocks"): break
    else:
        print("  "*d + f"{name}: {n!r}")
show(tree)
