import json, sys
from openalex import get

SEEDS = {
    "dreamer": "W2992977009",
    "dreamerv2": "W3091507139",
    "dreamerv3": "W4409147637",
    "ijepa": "W4386076428",
    "vjepa": "W4394861491",
    "vjepa2": "W4417261359",
    "rwm": "W2890208753",
    "worldmodels": "W2795843265",
    "planet": "W2900152462",
}

out = {}
for key, wid in SEEDS.items():
    d = get("/works", {
        "filter": f"cites:{wid},publication_year:>2022",
        "sort": "cited_by_count:desc",
        "per_page": 200,
        "page": 1,
        "select": "id,doi,title,publication_year,publication_date,cited_by_count,type,"
                  "authorships,primary_location,abstract_inverted_index,ids,referenced_works_count",
    }, f"cites_{key}")
    out[key] = d
    print(key, "total", d["meta"]["count"], "fetched", len(d["results"]),
          "top:", d["results"][0]["title"][:70] if d["results"] else None, flush=True)
json.dump(out, open("raw/cites_all.json", "w"))
