"""
Step 1: scrape publicly available pages from DarGlobal and Wasalt.

Usage:
    pip install requests beautifulsoup4 lxml
    python scraper.py                 # scrape both sites
    python scraper.py --site wasalt   # one site only
    python scraper.py --playwright    # use a headless browser if pages come back empty
                                      # (pip install playwright && playwright install chromium)

Output: data/darglobal.json and data/wasalt.json
        each is a list of {"source", "url", "title", "description", "text"}
"""
import argparse
import json
import re
import time
import urllib.robotparser
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 (compatible; AssignmentBot/1.0; +mailto:you@example.com)"  # put your email
DELAY = 1.0  # seconds between requests: be polite
OUT = Path("data")

SITES = {
    "darglobal": {
        "base": "https://darglobal.co.uk",
        "seeds": [
            "https://darglobal.co.uk/",
            "https://darglobal.co.uk/about",
            "https://darglobal.co.uk/investor",
            "https://darglobal.co.uk/get-in-touch",
        ],
        # skip other languages and noise
        "exclude": [r"/(ar|ru|zh|de|fr|es)(/|$)", r"/pay-online", r"/old-investor-page",
                    r"\.(pdf|jpg|png|webp|svg|zip)$"],
        "max_pages": 150,
        "caps": {},
    },
    "wasalt": {
        "base": "https://wasalt.sa",
        "seeds": [
            "https://wasalt.sa/en",
            "https://wasalt.sa/en/corporate",
            "https://wasalt.sa/en/properties-for-sale-in-saudi-arabia",
        ],
        # English only; skip Arabic slugs
        "include": [r"^https://wasalt\.sa/en(/|$)"],
        "exclude": [r"[\u0600-\u06FF]", r"%D[89AB]", r"\.(pdf|jpg|png|webp|svg|zip)$"],
        "max_pages": 300,
        # ~48k listings exist: sample, don't hoard
        "caps": {r"/en/property/": 150},
    },
}

session = requests.Session()
session.headers.update({"User-Agent": UA, "Accept-Language": "en"})


def get_robots(base):
    rp = urllib.robotparser.RobotFileParser()
    try:
        r = session.get(base + "/robots.txt", timeout=15)
        rp.parse(r.text.splitlines() if r.ok else [])
        sitemaps = re.findall(r"(?im)^sitemap:\s*(\S+)", r.text) if r.ok else []
    except requests.RequestException:
        rp.parse([])
        sitemaps = []
    return rp, sitemaps


def parse_sitemap(url, depth=0):
    """Return page URLs from a sitemap (follows sitemap indexes)."""
    if depth > 2:
        return []
    try:
        r = session.get(url, timeout=20)
        if not r.ok:
            return []
        root = ET.fromstring(r.content)
    except (requests.RequestException, ET.ParseError):
        return []
    locs = [e.text.strip() for e in root.iter() if e.tag.endswith("loc") and e.text]
    if root.tag.endswith("sitemapindex"):
        urls = []
        for loc in locs[:20]:
            urls += parse_sitemap(loc, depth + 1)
            time.sleep(0.3)
        return urls
    return locs


def allowed(url, cfg):
    if any(re.search(p, url) for p in cfg.get("exclude", [])):
        return False
    inc = cfg.get("include")
    return not inc or any(re.search(p, url) for p in inc)


def fetch_html(url, use_playwright=False, pw_page=None):
    if use_playwright and pw_page:
        pw_page.goto(url, wait_until="networkidle", timeout=45000)
        return pw_page.content()
    for attempt in range(3):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200 and "text/html" in r.headers.get("content-type", ""):
                return r.text
            if r.status_code in (403, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return None


def next_data_text(soup):
    """Next.js sites embed page data as JSON in <script id="__NEXT_DATA__">.
    Pull human-readable strings out of it (useful when HTML body is thin)."""
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string:
        return ""
    try:
        data = json.loads(tag.string)
    except json.JSONDecodeError:
        return ""
    out, seen = [], set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in {"buildId", "assetPrefix", "runtimeConfig", "query"}:
                    continue
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            s = re.sub(r"<[^>]+>", " ", o).strip()
            if len(s) > 25 and not s.startswith(("http", "/", "data:")) and s not in seen:
                seen.add(s)
                out.append(s)

    walk(data.get("props", {}).get("pageProps", {}))
    return "\n".join(out)


def extract(html, url):
    soup = BeautifulSoup(html, "lxml")
    nd = next_data_text(soup)
    for t in soup(["script", "style", "noscript", "svg", "iframe", "header", "footer", "nav", "form"]):
        t.decompose()
    title = (soup.title.string or "").strip() if soup.title else ""
    md = soup.find("meta", attrs={"name": "description"})
    desc = (md.get("content") or "").strip() if md else ""
    root = soup.find("main") or soup.body or soup
    lines = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th", "span", "div"]):
        # only leaf-ish blocks to avoid duplicated nested text
        if el.name in ("span", "div") and el.find(["p", "li", "div", "h1", "h2", "h3"]):
            continue
        s = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if len(s) > 2:
            lines.append(("# " if el.name in ("h1", "h2", "h3") else "") + s)
    # dedupe consecutive repeats
    clean = [l for i, l in enumerate(lines) if i == 0 or l != lines[i - 1]]
    text = "\n".join(clean)
    if len(text) < 400 and nd:  # body thin -> likely JS rendered, use embedded JSON
        text = (text + "\n" + nd).strip()
    links = {urldefrag(urljoin(url, a["href"]))[0].rstrip("/") for a in BeautifulSoup(html, "lxml").find_all("a", href=True)}
    return {"title": title, "description": desc, "text": text}, links


def scrape_site(name, cfg, use_playwright=False):
    base = cfg["base"]
    rp, robots_sitemaps = get_robots(base)
    print(f"\n=== {name} === robots sitemaps: {robots_sitemaps or 'none'}")

    # 1) discover URLs: seeds first, then sitemap, then crawl links
    queue = deque(cfg["seeds"])
    sitemap_urls = []
    for sm in robots_sitemaps or [base + "/sitemap.xml"]:
        sitemap_urls += parse_sitemap(sm)
    print(f"sitemap URLs found: {len(sitemap_urls)}")
    queue.extend(u.rstrip("/") for u in sitemap_urls if allowed(u, cfg))

    seen, results, cap_counts = set(), [], {}
    pw = browser = page = None
    if use_playwright:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        browser = pw.chromium.launch()
        page = browser.new_page(user_agent=UA)

    try:
        while queue and len(results) < cfg["max_pages"]:
            url = queue.popleft().rstrip("/")
            if url in seen or not url.startswith(base) or not allowed(url, cfg):
                continue
            seen.add(url)
            if not rp.can_fetch(UA, url):
                print("  robots.txt disallows:", url)
                continue
            capped = False
            for pat, limit in cfg["caps"].items():
                if re.search(pat, url):
                    if cap_counts.get(pat, 0) >= limit:
                        capped = True
                    cap_counts[pat] = cap_counts.get(pat, 0) + 1
            if capped:
                continue

            html = fetch_html(url, use_playwright, page)
            time.sleep(DELAY)
            if not html:
                continue
            doc, links = extract(html, url)
            if len(doc["text"]) < 80:
                print("  [thin]", url, len(doc["text"]), "chars")
                continue
            doc.update(source=name, url=url)
            results.append(doc)
            print(f"  [{len(results)}] {url} ({len(doc['text'])} chars)")
            for l in links:
                if l not in seen and l.startswith(base):
                    queue.append(l)
    finally:
        if browser:
            browser.close()
            pw.stop()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=list(SITES) + ["all"], default="all")
    ap.add_argument("--playwright", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for name, cfg in SITES.items():
        if args.site not in ("all", name):
            continue
        docs = scrape_site(name, cfg, args.playwright)
        (OUT / f"{name}.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
        total = sum(len(d["text"]) for d in docs)
        print(f"saved {len(docs)} pages, {total:,} chars -> data/{name}.json")


if __name__ == "__main__":
    main()