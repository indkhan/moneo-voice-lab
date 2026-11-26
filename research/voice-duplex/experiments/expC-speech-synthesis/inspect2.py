import sys, json, numpy as np
z = np.load(sys.argv[1])
tree = json.loads(str(z["__tree__"]))
print({k: (v if not isinstance(v,(dict,list)) else type(v).__name__) for k,v in tree.items()})
for k in ("max_frames","fps"): print(k, tree[k])
for name,v in tree["voices"].items(): print("voice", name, z[v["@"]].shape, z[v["@"]].dtype)
print("mean", z[tree["mean"]["@"]].shape)
def shp(w): return w["shape"] if "shape" in w else z[w["@"]].shape
for i,(kind,p) in enumerate(tree["decoder"]["seanet"]):
    if kind=="conv": print(i,kind,shp(p["w"]))
    elif kind=="convt": print(i,kind,shp(p["w"]),"stride",p["stride"])
    elif kind=="res": print(i,kind,shp(p[0]["w"]),shp(p[1]["w"]))
    else: print(i,kind)
