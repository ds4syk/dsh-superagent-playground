"""OpenAlex client with proxy-node rotation on 429 and local caching."""
import json, os, time, urllib.parse
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "raw")
os.makedirs(CACHE, exist_ok=True)

MIHOMO = "http://127.0.0.1:9090"
TOKEN = "a4504424a8a36021"
GROUP = "🔰 选择节点"
NODES = [
    "🇸🇬 新加坡W01 | IEPL | x2",
    "🇩🇪 德国W01",
    "🇬🇧 英国W01",
    "🇫🇷 法国W01",
    "🇨🇦 加拿大W01",
    "🇦🇺 澳大利亚W01",
    "🇰🇷 韩国W01",
    "🇺🇲 美国W02 | IEPL | x1.5",
]

def switch(node):
    try:
        requests.put(f"{MIHOMO}/proxies/{urllib.parse.quote(GROUP)}",
                     json={"name": node},
                     headers={"Authorization": f"Bearer {TOKEN}"}, timeout=5)
        time.sleep(1.0)
    except Exception as e:
        print("switch failed", e)

def get(path, params=None, cache_key=None, rotate=True):
    """GET api.openalex.org/<path> with cache and rotation."""
    if cache_key:
        fn = os.path.join(CACHE, cache_key + ".json")
        if os.path.exists(fn):
            return json.load(open(fn))
    url = "https://api.openalex.org" + path
    for attempt in range(len(NODES) * 2):
        try:
            r = requests.get(url, params=params, timeout=40,
                             headers={"User-Agent": "wm-citation-survey/1.0 (mailto:survey@example.org)"})
        except Exception as e:
            print("req error", e)
            time.sleep(3)
            continue
        if r.status_code == 200:
            data = r.json()
            if cache_key:
                json.dump(data, open(fn, "w"))
            time.sleep(0.6)
            return data
        if r.status_code == 429:
            print("429, rotating", attempt)
            if rotate:
                switch(NODES[attempt % len(NODES)])
                continue
        print("status", r.status_code, r.text[:200])
        time.sleep(2)
    raise RuntimeError("failed: " + path)

def search_title(title, key):
    return get("/works", {"filter": f'title.search:"{title}"', "per_page": 5,
                          "select": "id,doi,title,publication_year,cited_by_count,type,primary_location"}, key)

def cited_by(work_id, key, year_from=2023, per_page=200, page=1):
    return get("/works", {"filter": f"cites:{work_id},publication_year:>{year_from-1}",
                          "sort": "cited_by_count:desc", "per_page": per_page, "page": page,
                          "select": "id,doi,title,publication_year,cited_by_count,type,"
                                    "authorships,primary_location,abstract_inverted_index,ids"},
               key)
