#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

BASE = "https://iconographic.warburg.sas.ac.uk"
UA = "Voynich crown corpus research/0.1 (academic, low-rate)"


def atomic_json(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def parse_object(url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    out = {"object_url": url, "page_text": text}
    h1 = soup.find("h1")
    out["title"] = h1.get_text(" ", strip=True) if h1 else ""
    iiif = None
    for a in soup.find_all("a", href=True):
        if "iiif manifest" in a.get_text(" ", strip=True).lower():
            iiif = urljoin(url, a["href"])
            break
    out["iiif_manifest_url"] = iiif
    # Preserve raw text and extract only conservative labelled fields.
    for key, label in [
        ("artist", "Artist or creator"),
        ("date_label", "Date"),
        ("location", "Location"),
        ("book_text_document", "Book, text or document"),
        ("image_subject", "Image"),
    ]:
        m = re.search(rf"{re.escape(label)}\s*\n([^\n]+)", text, flags=re.I)
        out[key] = m.group(1).strip() if m else ""
    return out


def image_url_from_manifest(m: dict) -> str | None:
    # IIIF Presentation v3
    try:
        body = m["items"][0]["items"][0]["items"][0]["body"]
        if isinstance(body, list):
            body = body[0]
        service = body.get("service")
        if service:
            if isinstance(service, list):
                service = service[0]
            sid = service.get("id") or service.get("@id")
            if sid:
                return sid.rstrip("/") + "/full/1600,/0/default.jpg"
        return body.get("id") or body.get("@id")
    except Exception:
        pass
    # IIIF Presentation v2
    try:
        res = m["sequences"][0]["canvases"][0]["images"][0]["resource"]
        service = res.get("service")
        if service:
            if isinstance(service, list):
                service = service[0]
            sid = service.get("@id") or service.get("id")
            if sid:
                return sid.rstrip("/") + "/full/1600,/0/default.jpg"
        return res.get("@id") or res.get("id")
    except Exception:
        return None


async def fill_label(page, patterns, value):
    for pat in patterns:
        try:
            loc = page.get_by_label(re.compile(pat, re.I))
            if await loc.count():
                await loc.first.fill(value)
                return True
        except Exception:
            pass
    return False


async def discover_result_urls(query: str, earliest: int, latest: int, debug_dir: Path) -> list[str]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1500, "height": 1000})
        await page.goto(BASE + "/collections", wait_until="domcontentloaded", timeout=90000)
        try:
            t = page.get_by_text(re.compile("advanced search", re.I))
            if await t.count():
                await t.first.click(timeout=5000)
        except Exception:
            pass

        q_ok = await fill_label(page, ["general search", "search the database", "search"], query)
        e_ok = await fill_label(page, ["earliest year"], str(earliest))
        l_ok = await fill_label(page, ["latest year"], str(latest))

        if not q_ok:
            visible = page.locator('input[type="search"]:visible, input[type="text"]:visible')
            if await visible.count():
                await visible.first.fill(query)
                q_ok = True

        if not (q_ok and e_ok and l_ok):
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "search_form.html").write_text(await page.content(), encoding="utf-8")
            await page.screenshot(path=str(debug_dir / "search_form.png"), full_page=True)
            raise RuntimeError(f"Could not bind search controls: query={q_ok}, earliest={e_ok}, latest={l_ok}")

        clicked = False
        for pat in [r"^search$", r"submit"]:
            try:
                b = page.get_by_role("button", name=re.compile(pat, re.I))
                if await b.count():
                    await b.first.click()
                    clicked = True
                    break
            except Exception:
                pass
        if not clicked:
            await page.keyboard.press("Enter")
        await page.wait_for_load_state("domcontentloaded", timeout=90000)

        urls: list[str] = []
        seen_sigs = set()
        for _ in range(1000):
            hrefs = await page.locator('a[href*="/object-wpc-wid-"]').evaluate_all("els => els.map(e => e.href)")
            hrefs = sorted(set(hrefs))
            sig = hashlib.sha1((page.url + "\n" + "\n".join(hrefs)).encode()).hexdigest()
            if sig in seen_sigs:
                break
            seen_sigs.add(sig)
            urls.extend(hrefs)

            next_link = None
            for pat in [r"^next$", r"next", r"^>$", r"^›$"]:
                try:
                    cand = page.get_by_role("link", name=re.compile(pat, re.I))
                    if await cand.count():
                        next_link = cand.first
                        break
                except Exception:
                    pass
            if next_link is None:
                break
            try:
                await next_link.click(timeout=10000)
                await page.wait_for_load_state("domcontentloaded", timeout=90000)
            except Exception:
                break
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "result_page_final.html").write_text(await page.content(), encoding="utf-8")
        await page.screenshot(path=str(debug_dir / "result_page_final.png"), full_page=True)
        await browser.close()
    return sorted(set(urls))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="crown")
    ap.add_argument("--earliest", type=int, default=1390)
    ap.add_argument("--latest", type=int, default=1440)
    ap.add_argument("--out", type=Path, default=Path("crown_corpus/output"))
    ap.add_argument("--max-results", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    out = args.out
    raw = out / "raw"
    images = out / "images"
    debug = out / "debug"
    for d in [out, raw, images, debug]:
        d.mkdir(parents=True, exist_ok=True)

    state_path = out / "harvest_state.json"
    state = {"query": args.query, "earliest": args.earliest, "latest": args.latest,
             "result_urls": [], "objects": {}, "failures": []}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    if not state["result_urls"]:
        state["result_urls"] = await discover_result_urls(args.query, args.earliest, args.latest, debug)
        if args.max_results:
            state["result_urls"] = state["result_urls"][:args.max_results]
        atomic_json(state, state_path)

    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    for idx, url in enumerate(state["result_urls"], 1):
        if url in state["objects"]:
            continue
        rec = {"object_url": url, "status": "started"}
        try:
            r = sess.get(url, timeout=45)
            r.raise_for_status()
            rec.update(parse_object(url, r.text))
            digest = hashlib.sha1(url.encode()).hexdigest()[:16]
            (raw / f"{digest}.html").write_text(r.text, encoding="utf-8")

            if rec.get("iiif_manifest_url"):
                mr = sess.get(rec["iiif_manifest_url"], timeout=45)
                mr.raise_for_status()
                mani = mr.json()
                (raw / f"{digest}.iiif.json").write_text(json.dumps(mani, ensure_ascii=False), encoding="utf-8")
                rec["image_url"] = image_url_from_manifest(mani)
            else:
                rec["image_url"] = None

            if rec.get("image_url"):
                ir = sess.get(rec["image_url"], timeout=90)
                ir.raise_for_status()
                ipath = images / f"{digest}.jpg"
                ipath.write_bytes(ir.content)
                rec["image_path"] = str(ipath)
                rec["image_bytes"] = len(ir.content)
                rec["status"] = "downloaded"
            else:
                rec["status"] = "no_image_url"
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = repr(exc)
            state["failures"].append({"object_url": url, "error": repr(exc)})
        state["objects"][url] = rec
        atomic_json(state, state_path)
        time.sleep(0.15)

    compact = list(state["objects"].values())
    atomic_json(compact, out / "manifest.json")
    print(json.dumps({
        "warburg_result_urls": len(state["result_urls"]),
        "object_records": len(compact),
        "downloaded_images": sum(r.get("status") == "downloaded" for r in compact),
        "failures": len(state["failures"]),
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
