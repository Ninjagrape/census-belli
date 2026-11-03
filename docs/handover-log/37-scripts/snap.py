"""Snapshot passage hashes and infobox quantities for every article under data/raw."""
import dataclasses, hashlib, json, sys
from pathlib import Path
from pipeline.extractors.article import build_passages
from pipeline.extractors.infobox import parse_article_infobox

raw_dir = Path(sys.argv[1]); out = Path(sys.argv[2])
res = {"passages": {}, "infobox": {}}
for p in sorted((raw_dir / "battles_html").iterdir()):
    if p.suffix not in (".html", ".wikitext"):
        continue
    raw = p.read_text(encoding="utf-8", errors="replace")
    is_html = p.suffix == ".html"
    ps = build_passages(p.stem, raw, source_title=p.stem, is_html=is_html)
    res["passages"][p.name] = [hashlib.sha256(json.dumps(dataclasses.asdict(x), sort_keys=True).encode()).hexdigest() for x in ps]
    try:
        ex = parse_article_infobox(raw, is_html=is_html, source_ref=p.name, source_title=p.stem)
    except Exception as e:
        res["infobox"][p.name] = f"ERR {type(e).__name__}"
        continue
    if ex is None:
        res["infobox"][p.name] = None
        continue
    res["infobox"][p.name] = [
        [s.label,
         [(r.reported_value, r.branch, r.is_lower_bound, r.is_upper_bound) for r in s.troop_reports],
         [(r.reported_value, r.casualty_type) for r in s.casualty_reports]]
        for s in ex.sides]
out.write_text(json.dumps(res))
