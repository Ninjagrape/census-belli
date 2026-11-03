import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
print("files:", len(a["passages"]))
d = [k for k in a["passages"] if a["passages"][k] != b["passages"][k]]
print("passage-hash differing articles:", len(d), d[:10])
print("total passages:", sum(len(v) for v in a["passages"].values()))
ch = [k for k in a["infobox"] if a["infobox"][k] != b["infobox"][k]]
print("infobox changes:", len(ch))
for k in ch:
    print("==", k)
    A, B = a["infobox"][k], b["infobox"][k]
    if not isinstance(A, list) or not isinstance(B, list):
        print("  old:", A, "\n  new:", B); continue
    for i in range(max(len(A), len(B))):
        x = A[i] if i < len(A) else None; y = B[i] if i < len(B) else None
        if x != y:
            print(f"  side{i} old:", x); print(f"  side{i} new:", y)
