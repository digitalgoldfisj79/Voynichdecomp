#!/usr/bin/env python3
"""Backfill ellievelinska.blogspot.com from the Wayback Machine.

Usage:
    python backfill.py all            # enumerate -> fetch -> parse -> verify -> zip
    python backfill.py enumerate|fetch|parse|verify|zip

Every stage is resumable. State lives in manifest.json (written atomically
after each fetch). Nothing is uploaded anywhere; the only network traffic is
GET requests to web.archive.org.
"""
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from urllib.parse import urlsplit, parse_qsl

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

HOST = "ellievelinska.blogspot.com"
# Blogger served country-code redirects (e.g. .co.uk) for years; Wayback indexes
# those under separate urlkeys, so query them too. Missing hosts just return [].
EXTRA_HOSTS = ["ellievelinska.blogspot.co.uk", "ellievelinska.blogspot.com.au",
               "ellievelinska.blogspot.ca", "ellievelinska.blogspot.ie"]
CDX = "https://web.archive.org/cdx/search/cdx"
WB = "https://web.archive.org/web/{ts}id_/{url}"
DELAY = 2.0
RETRIES = 5
UA = "velinska-backfill/1.0 (personal archive recovery; python-requests)"

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "raw")
FEEDS = os.path.join(ROOT, "feeds")
MONTHS = os.path.join(ROOT, "raw", "_month_pages")
MANIFEST = os.path.join(ROOT, "manifest.json")
CDX_CACHE = os.path.join(ROOT, "cdx_rows.json")

EXPECTED_SPEC = """2010-12:19 | 2011-01:2 05:1 07:1 10:1 | 2012-04:3 06:1 07:4 08:1 10:2 12:1 |
2013-01:6 02:3 03:15 04:23 05:24 06:18 07:21 08:2 09:8 10:1 11:2 |
2014-01:3 02:2 03:2 05:1 06:1 09:1 10:2 11:1 | 2015-02:4 04:3 | 2016-01:1 03:1 04:1 10:1 11:1"""

POST_RE = re.compile(r"^/(\d{4})/(\d{2})/([^/]+)\.html$")
MONTH_RE = re.compile(r"^/(\d{4})/(\d{2})/?$")
OLD_ARCHIVE_RE = re.compile(r"^/(\d{4})_(\d{2})_\d{2}_archive\.html$")
FEED_RE = re.compile(r"^/feeds/posts/(default|summary)")


def expected_counts():
    out = {}
    for group in EXPECTED_SPEC.replace("\n", " ").split("|"):
        year = None
        for tok in group.split():
            if "-" in tok:
                year, tok = tok.split("-", 1)
            m, n = tok.split(":")
            out[f"{year}-{m}"] = int(n)
    return out


# ---------------------------------------------------------------- network

_last = [0.0]


def get(url, params=None, timeout=60):
    """Rate-limited GET (1 request / DELAY s) with exponential backoff."""
    for attempt in range(RETRIES + 1):
        wait = _last[0] + DELAY - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": UA})
        except requests.RequestException as e:
            err = f"{type(e).__name__}: {e}"
            r = None
        if r is not None:
            if r.status_code == 429 or r.status_code >= 500:
                err = f"HTTP {r.status_code}"
            else:
                return r
        if attempt == RETRIES:
            raise RuntimeError(f"giving up on {url}: {err}")
        back = DELAY * 2 ** (attempt + 1)
        print(f"  retry {attempt + 1}/{RETRIES} in {back:.0f}s ({err})", flush=True)
        time.sleep(back)


# ---------------------------------------------------------------- manifest

def load_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            return json.load(f)
    return {"posts": {}, "month_pages": {}, "feeds": {}, "cdx": {}}


def save_manifest(m):
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=1, sort_keys=True)
    os.replace(tmp, MANIFEST)


# ---------------------------------------------------------------- 1. enumerate

def cdx_rows(host, collapse):
    """All 200 captures under host/*. Pages through CDX with resumeKey."""
    params = {"url": f"{host}/*", "output": "json",
              "fl": "timestamp,original,statuscode,mimetype",
              "filter": "statuscode:200", "showResumeKey": "true", "limit": "5000"}
    if collapse:
        params["collapse"] = "urlkey"
    rows, resume = [], None
    while True:
        if resume:
            params["resumeKey"] = resume
        r = get(CDX, params=params, timeout=180)
        if r.status_code != 200:
            raise RuntimeError(f"CDX {host}: HTTP {r.status_code}: {r.text[:200]}")
        data = r.json() if r.text.strip() else []
        resume = None
        if data and data[-1] and len(data[-1]) == 1:   # [resumeKey] after a blank []
            resume = data[-1][0]
            data = [d for d in data[:-1] if d]
        rows += [d for d in data[1:] if d]               # drop header row
        if not resume:
            return rows


def classify(original):
    """Return (kind, key, variant) for a captured URL, or None."""
    u = urlsplit(original)
    host = (u.hostname or "").lower()
    if not host.startswith("ellievelinska.blogspot."):
        return None
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    path = u.path
    if m := POST_RE.match(path):
        variant = "direct"
        if "m" in q:
            variant = "mobile"
        elif "showComment" in q:
            variant = "showComment"
        elif q:
            variant = "query"
        return "post", f"{m[1]}-{m[2]}-{m[3]}", variant
    if m := MONTH_RE.match(path) or OLD_ARCHIVE_RE.match(path):
        return "month", f"{m[1]}-{m[2]}", "mobile" if "m" in q else "direct"
    if FEED_RE.match(path):
        return "feed", path + ("?" + u.query if u.query else ""), "feed"
    return None


def canonical(key):
    y, mo, slug = key.split("-", 2)
    return f"http://{HOST}/{y}/{mo}/{slug}.html"


def enumerate_stage():
    m = load_manifest()
    if os.path.exists(CDX_CACHE):
        print("CDX cache exists; reusing (delete cdx_rows.json to refresh)")
        with open(CDX_CACHE) as f:
            cache = json.load(f)
    else:
        cache = {}
        for host in [HOST] + EXTRA_HOSTS:
            # The spec query uses collapse=urlkey, which keeps the FIRST capture per
            # urlkey. We also need the LATEST, so take the uncollapsed listing as
            # the source of truth and record the collapsed count for reference.
            print(f"CDX {host} (collapsed)")
            collapsed = cdx_rows(host, True)
            print(f"  {len(collapsed)} urlkeys")
            if not collapsed:
                cache[host] = {"collapsed": 0, "rows": []}
                continue
            print(f"CDX {host} (all captures)")
            rows = cdx_rows(host, False)
            print(f"  {len(rows)} captures")
            cache[host] = {"collapsed": len(collapsed), "rows": rows}
        with open(CDX_CACHE, "w") as f:
            json.dump(cache, f)

    posts = defaultdict(lambda: defaultdict(list))
    months = defaultdict(list)
    feeds = {}
    for host, blob in cache.items():
        m["cdx"][host] = {"urlkeys_collapsed": blob["collapsed"],
                          "captures": len(blob["rows"])}
        for ts, orig, status, mime in blob["rows"]:
            c = classify(orig)
            if not c:
                continue
            kind, key, variant = c
            if kind == "post":
                posts[key][variant].append((ts, orig))
            elif kind == "month":
                months[key].append((ts, orig, variant))
            elif kind == "feed":
                feeds.setdefault(f"{ts}|{orig}", {"timestamp": ts, "original": orig,
                                                  "mimetype": mime})

    for key, variants in posts.items():
        entry = m["posts"].setdefault(key, {"status": "pending"})
        entry["url"] = canonical(key)
        # candidates in fallback order; latest capture first within each variant
        cands = []
        for v in ("direct", "query", "showComment", "mobile"):
            for ts, orig in sorted(variants.get(v, []), reverse=True):
                cands.append({"variant": v, "timestamp": ts, "original": orig})
        entry["candidates"] = cands
    for key, caps in months.items():
        entry = m["month_pages"].setdefault(key, {"status": "pending"})
        caps = sorted(caps, key=lambda c: (c[2] != "direct", -int(c[0])))
        entry["candidates"] = [{"timestamp": t, "original": o, "variant": v}
                               for t, o, v in caps]
    for k, f in feeds.items():
        m["feeds"].setdefault(k, {**f, "status": "pending"})
    save_manifest(m)
    print(f"enumerated {len(posts)} post keys, {len(months)} month pages, "
          f"{len(feeds)} feed captures")


# ---------------------------------------------------------------- 2. fetch

def fetch_one(cand, path):
    url = WB.format(ts=cand["timestamp"], url=cand["original"])
    r = get(url)
    if r.status_code != 200 or not r.content:
        return f"HTTP {r.status_code}"
    with open(path, "wb") as f:
        f.write(r.content)
    return None


def try_candidates(entry, path, m, label):
    for cand in entry.get("candidates", []):
        tried = entry.setdefault("tried", [])
        tag = f"{cand['timestamp']}|{cand['original']}"
        if tag in tried:
            continue
        print(f"  {label} <- {cand['timestamp']} {cand['original']}", flush=True)
        try:
            err = fetch_one(cand, path)
        except RuntimeError as e:
            err = str(e)
        tried.append(tag)
        if err is None:
            entry.update(status="fetched", file=os.path.relpath(path, ROOT),
                         capture_timestamp=cand["timestamp"],
                         fetched_from=cand["original"], variant=cand["variant"])
            save_manifest(m)
            return True
        entry["last_error"] = err
        save_manifest(m)
    return False


def fetch_stage():
    m = load_manifest()
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(MONTHS, exist_ok=True)
    os.makedirs(FEEDS, exist_ok=True)

    # feeds first: cheap, and they tell us titles for missing posts
    for k, f in sorted(m["feeds"].items()):
        if f["status"] == "fetched":
            continue
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", f["original"].split("/feeds/", 1)[1])
        path = os.path.join(FEEDS, f"{f['timestamp']}_{safe}.xml")
        print(f"feed {f['timestamp']} {f['original']}")
        try:
            err = fetch_one(f, path)
        except RuntimeError as e:
            err = str(e)
        if err is None:
            f.update(status="fetched", file=os.path.relpath(path, ROOT))
        else:
            f.update(status="failed", last_error=err)
        save_manifest(m)

    keys = sorted(m["posts"])
    for i, key in enumerate(keys, 1):
        e = m["posts"][key]
        if e["status"] == "fetched":
            continue
        path = os.path.join(RAW, f"{key}.html")
        if not try_candidates(e, path, m, f"[{i}/{len(keys)}] {key}"):
            e["status"] = "no_capture"
            save_manifest(m)

    # month pages: needed for posts with no capture, plus any month short of its
    # expected count (month pages can name posts the CDX never saw)
    exp = expected_counts()
    have = Counter(k[:7] for k, e in m["posts"].items() if e["status"] == "fetched")
    want = {k[:7] for k, e in m["posts"].items() if e["status"] != "fetched"}
    want |= {mo for mo, n in exp.items() if have[mo] < n}
    for mo in sorted(want):
        e = m["month_pages"].get(mo)
        if not e:
            m["month_pages"][mo] = {"status": "no_capture", "candidates": []}
            continue
        if e["status"] == "fetched":
            continue
        path = os.path.join(MONTHS, f"{mo}.html")
        if not try_candidates(e, path, m, f"month {mo}"):
            e["status"] = "no_capture"
        save_manifest(m)
    print("fetch done")


# ---------------------------------------------------------------- 3. parse

BLOCK = {"p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
         "blockquote", "table", "tr", "pre", "section", "article", "figure"}


def block_text(node):
    parts = []

    def walk(n):
        if isinstance(n, NavigableString):
            if n.__class__.__name__ in ("Comment", "Script", "Stylesheet"):
                return
            parts.append(str(n))
        elif isinstance(n, Tag):
            if n.name in ("script", "style", "noscript"):
                return
            if n.name == "br":
                parts.append("\n")
                return
            blk = n.name in BLOCK
            if blk:
                parts.append("\n")
            for c in n.children:
                walk(c)
            if blk:
                parts.append("\n")

    walk(node)
    text = "".join(parts).replace("\xa0", " ")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def unwayback(u):
    """Strip a Wayback prefix if one slipped through (id_ shouldn't add any)."""
    if not u:
        return u
    m = re.match(r"^(?:https?://web\.archive\.org)?/web/\d+[a-z_]*/(.+)$", u)
    return m.group(1) if m else u


def parse_post_node(node, soup_for_date=None):
    """Extract fields from a div.post (or a whole page containing one)."""
    title_el = node.select_one("h3.post-title") or node.select_one(".post-title")
    title = block_text(title_el) if title_el else None
    link = title_el.find("a") if title_el else None
    published = None
    ab = node.select_one("abbr.published")
    if ab and ab.get("title"):
        published = ab["title"].strip()
    else:
        t = node.select_one("time.published")
        if t and t.get("datetime"):
            published = t["datetime"].strip()
    if not published and soup_for_date is not None:
        dh = soup_for_date.select_one("h2.date-header")
        if dh:
            published = block_text(dh)
    if not published:
        lk = node.find("a", class_="timestamp-link")
        if lk:
            published = block_text(lk)
    labels = []
    for sel in (".post-labels a", "span.post-labels a", ".labels a"):
        labels = [block_text(a) for a in node.select(sel)]
        if labels:
            break
    body_el = node.select_one("div.post-body") or node.select_one(".post-body")
    body = block_text(body_el) if body_el else None
    images = []
    if body_el:
        for img in body_el.find_all("img"):
            src = unwayback(img.get("src") or img.get("data-src"))
            if src:
                images.append({"src": src, "alt": img.get("alt", "")})
    return {"title": title, "href": unwayback(link.get("href")) if link else None,
            "published": published, "labels": labels, "body_text": body,
            "images": images}


def parse_comments(soup, post_url):
    out = []
    # classic layout: dl#comments-block > dt.comment-author / dd.comment-body / dd.comment-footer
    blk = soup.select_one("#comments-block")
    if blk:
        for dt in blk.find_all("dt", class_="comment-author"):
            body = dt.find_next_sibling("dd", class_="comment-body")
            foot = dt.find_next_sibling("dd", class_="comment-footer")
            a = dt.find("a", rel=lambda r: r and "nofollow" in r) or dt.find("a")
            author = block_text(a) if a else re.sub(r"\s*said\.*:?\s*$", "", block_text(dt))
            ts = foot.select_one(".comment-timestamp") if foot else None
            out.append({"post_url": post_url, "author": author,
                        "date": block_text(ts) if ts else (block_text(foot) if foot else None),
                        "text": block_text(body) if body else ""})
        return out
    # threaded layout: .comment-thread li.comment
    for li in soup.select(".comment-thread li.comment, .comments li.comment"):
        cite = li.select_one("cite.user") or li.select_one(".user")
        dt = li.select_one(".datetime") or li.select_one(".comment-timestamp")
        body = li.select_one(".comment-content")
        if body is None:
            continue
        out.append({"post_url": post_url,
                    "author": block_text(cite) if cite else None,
                    "date": block_text(dt) if dt else None,
                    "text": block_text(body)})
    return out


def soup_of(path):
    with open(path, "rb") as f:
        return BeautifulSoup(f.read(), "html.parser")


def key_from_href(href):
    if not href:
        return None
    u = urlsplit(unwayback(href))
    m = POST_RE.match(u.path)
    return f"{m[1]}-{m[2]}-{m[3]}" if m else None


def parse_feeds(m):
    """Return {key: {title, published, content_html, feed_file}} from saved feeds."""
    import xml.etree.ElementTree as ET
    A = "{http://www.w3.org/2005/Atom}"
    entries = {}
    totals = []
    for k, f in m["feeds"].items():
        if f.get("status") != "fetched":
            continue
        p = os.path.join(ROOT, f["file"])
        try:
            tree = ET.parse(p).getroot()
        except ET.ParseError as e:
            f["parse_error"] = str(e)
            continue
        tot = tree.find("{http://a9.com/-/spec/opensearchrss/1.0/}totalResults")
        if tot is None:
            tot = tree.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
        if tot is not None and tot.text:
            totals.append((f["timestamp"], int(tot.text), f["original"]))
        for e in tree.iter(A + "entry"):
            href = None
            for ln in e.findall(A + "link"):
                if ln.get("rel") == "alternate":
                    href = ln.get("href")
            key = key_from_href(href)
            if not key:
                continue
            cont = e.find(A + "content")
            rec = {"title": (e.findtext(A + "title") or "").strip(),
                   "published": e.findtext(A + "published"),
                   "labels": [c.get("term") for c in e.findall(A + "category")],
                   "content_html": cont.text if cont is not None else None,
                   "feed_file": f["file"], "timestamp": f["timestamp"]}
            # keep the entry from the latest feed capture that has content
            old = entries.get(key)
            if not old or (rec["content_html"] and rec["timestamp"] >= old["timestamp"]) \
                    or not old["content_html"]:
                entries[key] = rec
    return entries, totals


def parse_stage():
    m = load_manifest()
    posts, comments, failures = {}, [], []

    for key, e in sorted(m["posts"].items()):
        if e["status"] != "fetched":
            continue
        y, mo, slug = key.split("-", 2)
        try:
            soup = soup_of(os.path.join(ROOT, e["file"]))
            nodes = soup.select("div.post")
            node = nodes[0] if nodes else soup
            rec = parse_post_node(node, soup)
            if not rec["body_text"]:
                raise ValueError("no div.post-body")
        except Exception as ex:
            failures.append({"key": key, "file": e["file"], "error": str(ex)})
            e["parse_error"] = str(ex)
            continue
        rec.pop("href")
        posts[key] = {"url": e["url"], "slug": slug, "year": int(y), "month": int(mo),
                      **rec, "word_count": len(rec["body_text"].split()),
                      "capture_timestamp": e["capture_timestamp"],
                      "source": "mobile" if e["variant"] == "mobile" else "post"}
        comments += parse_comments(soup, e["url"])

    # month pages: fill posts not yet recovered, and list every title seen
    month_titles = {}
    for mo, e in sorted(m["month_pages"].items()):
        if e.get("status") != "fetched":
            continue
        try:
            soup = soup_of(os.path.join(ROOT, e["file"]))
        except Exception as ex:
            failures.append({"key": f"month {mo}", "file": e["file"], "error": str(ex)})
            continue
        for node in soup.select("div.post"):
            # date headers sit outside div.post, above a group of posts
            dh = node.find_previous("h2", class_="date-header")
            rec = parse_post_node(node)
            if not rec["published"] and dh:
                rec["published"] = block_text(dh)
            key = key_from_href(rec["href"])
            if not key:
                failures.append({"key": f"month {mo}", "file": e["file"],
                                 "error": f"post without permalink: {rec['title']!r}"})
                continue
            month_titles[key] = rec["title"]
            if key in posts or not rec["body_text"]:
                continue
            y, mm, slug = key.split("-", 2)
            rec.pop("href")
            posts[key] = {"url": canonical(key), "slug": slug, "year": int(y),
                          "month": int(mm), **rec,
                          "word_count": len(rec["body_text"].split()),
                          "capture_timestamp": e["capture_timestamp"],
                          "source": "month_page"}

    feed_entries, feed_totals = parse_feeds(m)
    for key, fe in feed_entries.items():
        if key in posts or not fe["content_html"]:
            continue
        body = BeautifulSoup(fe["content_html"], "html.parser")
        y, mm, slug = key.split("-", 2)
        text = block_text(body)
        posts[key] = {"url": canonical(key), "slug": slug, "year": int(y), "month": int(mm),
                      "title": fe["title"], "published": fe["published"],
                      "labels": fe["labels"], "body_text": text,
                      "images": [{"src": i.get("src"), "alt": i.get("alt", "")}
                                 for i in body.find_all("img") if i.get("src")],
                      "word_count": len(text.split()),
                      "capture_timestamp": fe["timestamp"], "source": "feed"}

    with open(os.path.join(ROOT, "posts.jsonl"), "w") as f:
        for key in sorted(posts):
            f.write(json.dumps(posts[key], ensure_ascii=False) + "\n")
    with open(os.path.join(ROOT, "comments.jsonl"), "w") as f:
        for c in comments:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    m["parse"] = {"failures": failures, "feed_totals": feed_totals,
                  "month_titles": month_titles,
                  "feed_index": {k: {"title": v["title"], "published": v["published"]}
                                 for k, v in feed_entries.items()}}
    save_manifest(m)
    print(f"parsed {len(posts)} posts, {len(comments)} comments, "
          f"{len(failures)} failures")


# ---------------------------------------------------------------- 4. verify

def verify_stage():
    m = load_manifest()
    p = m.get("parse", {})
    posts = {}
    with open(os.path.join(ROOT, "posts.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            posts[f"{r['year']}-{r['month']:02d}-{r['slug']}"] = r
    exp = expected_counts()
    found = Counter(k[:7] for k in posts)
    total_exp = sum(exp.values())
    L = ["# Coverage: ellievelinska.blogspot.com", "",
         f"Expected total (2020-05-23 archive widget): **{total_exp}**  ",
         f"Recovered: **{len(posts)} / {total_exp}**  ",
         "Source breakdown: " + ", ".join(f"{s}={n}" for s, n in
                                          Counter(r["source"] for r in posts.values()).most_common()),
         "", "## Per month", "", "| month | found | expected | diff |", "|---|---|---|---|"]
    for mo in sorted(set(exp) | set(found)):
        d = found[mo] - exp.get(mo, 0)
        flag = "" if d == 0 else (" ⚠" if d < 0 else " (extra)")
        L.append(f"| {mo} | {found[mo]} | {exp.get(mo, 0)} | {d:+d}{flag} |")

    known = {}
    for k, v in p.get("feed_index", {}).items():
        known[k] = (v["title"], "feed")
    for k, t in p.get("month_titles", {}).items():
        known.setdefault(k, (t, "month page"))
    for k, e in m["posts"].items():
        known.setdefault(k, (None, "CDX only"))
    missing = sorted(k for k in known if k not in posts)
    L += ["", f"## Missing posts ({len(missing)} named; "
              f"{max(0, total_exp - len(posts) - len(missing))} more unnamed vs widget count)", ""]
    if missing:
        L += ["| key | title | known from |", "|---|---|---|"]
        for k in missing:
            t, src = known[k]
            L.append(f"| {k} | {t or '(title unknown)'} | {src} |")
    else:
        L.append("None named.")

    L += ["", "## Feed cross-check", ""]
    tots = p.get("feed_totals", [])
    if not tots:
        L.append("No /feeds/posts/default capture was fetched; no feed cross-check.")
    for ts, n, url in tots:
        L.append(f"- feed {ts} ({url}): totalResults={n} (vs {total_exp} expected, "
                 f"{len(posts)} recovered)")
    mism = []
    for k, v in p.get("feed_index", {}).items():
        r = posts.get(k)
        if not r or r["source"] == "feed":
            continue
        if (v["title"] or "").strip() != (r["title"] or "").strip():
            mism.append(f"- {k}: title feed={v['title']!r} page={r['title']!r}")
        if v["published"] and r["published"] and v["published"][:10] != r["published"][:10] \
                and re.match(r"\d{4}-", r["published"]):
            mism.append(f"- {k}: date feed={v['published']} page={r['published']}")
    if p.get("feed_index"):
        L.append(f"Feed entries: {len(p['feed_index'])}; title/date mismatches: {len(mism)}")
        L += mism

    L += ["", "## Parse / fetch failures", ""]
    fails = p.get("failures", [])
    nocap = sorted(k for k, e in m["posts"].items() if e["status"] != "fetched")
    for f_ in fails:
        L.append(f"- parse: {f_['key']} ({f_['file']}): {f_['error']}")
    for k in nocap:
        if k not in posts:
            L.append(f"- fetch: {k}: {m['posts'][k].get('last_error', 'no capture')}")
    if not fails and not nocap:
        L.append("None.")
    no_date = [k for k, r in posts.items() if not r["published"]]
    no_title = [k for k, r in posts.items() if not r["title"]]
    if no_date or no_title:
        L += ["", f"Posts with no title: {len(no_title)} {no_title}",
              f"Posts with no published date: {len(no_date)} {no_date}"]
    with open(os.path.join(ROOT, "coverage.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"coverage.md: {len(posts)}/{total_exp}")


# ---------------------------------------------------------------- 5. zip

def zip_stage():
    out = os.path.join(ROOT, "velinska_backfill.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("posts.jsonl", "comments.jsonl", "coverage.md", "manifest.json"):
            p = os.path.join(ROOT, name)
            if os.path.exists(p):
                z.write(p, name)
        for d in ("feeds", "raw"):
            for dp, _, fs in os.walk(os.path.join(ROOT, d)):
                for fn in sorted(fs):
                    p = os.path.join(dp, fn)
                    z.write(p, os.path.relpath(p, ROOT))
    print(f"wrote {out}")


STAGES = {"enumerate": enumerate_stage, "fetch": fetch_stage, "parse": parse_stage,
          "verify": verify_stage, "zip": zip_stage}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    order = list(STAGES) if which == "all" else [which]
    for s in order:
        print(f"== {s}")
        STAGES[s]()
