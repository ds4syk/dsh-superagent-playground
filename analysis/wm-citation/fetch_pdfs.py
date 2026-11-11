import json, os, re, subprocess, sys, time
import requests
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
PDFS = os.path.join(ROOT, "pdfs")
TXT = os.path.join(ROOT, "txt")
os.makedirs(PDFS, exist_ok=True)
os.makedirs(TXT, exist_ok=True)

sel = json.load(open(os.path.join(ROOT, "raw/selected.json")))
ARXIV_API = "http://export.arxiv.org/api/query"

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def arxiv_from_doi(w):
    doi = (w.get("doi") or "").lower()
    m = re.search(r"10\.48550/arxiv\.([0-9]{4}\.[0-9]{4,5})", doi)
    if m:
        return m.group(1)
    return None

def arxiv_search(title):
    q = 'ti:"%s"' % title.replace('"', "")
    try:
        r = requests.get(ARXIV_API, params={"search_query": q, "max_results": 5}, timeout=40)
        entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.S)
        best, best_score = None, 0
        for e in entries:
            t = re.search(r"<title>(.*?)</title>", e, re.S)
            idm = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", e)
            if not t or not idm:
                continue
            tt = norm(re.sub(r"\s+", " ", t.group(1)))
            a, b = norm(title), tt
            # token overlap
            sa, sb = set(a.split()), set(b.split())
            score = len(sa & sb) / max(1, len(sa | sb))
            if score > best_score:
                best_score, best = score, idm.group(1)
        if best_score > 0.75:
            return best
    except Exception as ex:
        print("arxiv search fail", title[:50], ex)
    return None

def resolve(w):
    aid = arxiv_from_doi(w)
    if not aid:
        aid = arxiv_search(w["title"])
    return aid

def download(args):
    w, aid = args
    if not aid:
        return w["id"], None
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", aid)[:80]
    pdf = os.path.join(PDFS, safe + ".pdf")
    txt = os.path.join(TXT, safe + ".txt")
    if os.path.exists(txt) and os.path.getsize(txt) > 5000:
        return w["id"], txt
    if not os.path.exists(pdf) or os.path.getsize(pdf) < 20000:
        url = f"https://arxiv.org/pdf/{aid}"
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=90)
                if r.status_code == 200 and r.content[:4] == b"%PDF":
                    open(pdf, "wb").write(r.content)
                    break
            except Exception as e:
                print("dl fail", aid, e)
            time.sleep(2)
    if os.path.exists(pdf):
        subprocess.run(["pdftotext", "-q", pdf, txt], check=False)
    return w["id"], txt if os.path.exists(txt) else None

res = {}
with ThreadPoolExecutor(max_workers=4) as ex:
    ids = list(ex.map(resolve, sel))
for w, aid in zip(sel, ids):
    res[w["id"]] = aid
print("resolved", sum(1 for v in res.values() if v), "/", len(res))
missing = [w["title"] for w in sel if not res[w["id"]]]
for m in missing:
    print("NO ARXIV:", m[:90])
json.dump(res, open(os.path.join(ROOT, "raw/arxiv_ids.json"), "w"))

with ThreadPoolExecutor(max_workers=4) as ex:
    out = list(ex.map(download, [(w, res[w["id"]]) for w in sel]))
ok = {k: v for k, v in out if v}
print("texts", len(ok), "/", len(sel))
json.dump(ok, open(os.path.join(ROOT, "raw/texts.json"), "w"))
