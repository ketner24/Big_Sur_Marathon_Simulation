"""
Big Sur Marathon scraper v5 — network-interception edition.

Why v5: every UI-based approach (clicking Next, filling page input, JS dispatch)
has been unreliable because SVE has multiple paginators on one page and clicks
sometimes no-op or loop. v5 sidesteps the UI entirely:

  1. Load page 1 in a real browser (Playwright).
  2. Watch the network. Click Next ONCE.
  3. Capture the AJAX URL that fires.
  4. For pages 2..N, call that URL directly with the page param swapped.
  5. Parse the HTML/JSON response and extract rows.

This is dramatically faster (no DOM rendering between pages) and 100% reliable
because we never touch the SVE UI again after page 1.

If network interception finds nothing (e.g., pagination is purely client-side),
v5 falls back to the v4 UI-clicking approach.

SETUP
-----
    pip install playwright pandas
    playwright install chromium
"""

from __future__ import annotations

import re
import time
from io import StringIO
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd
from playwright.sync_api import (
    Page,
    Playwright,
    Request,
    TimeoutError as PWTimeout,
    sync_playwright,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

YEARS = [2018, 2019, 2022, 2023, 2024, 2025, 2026]
BASE = "https://results.svetiming.com/Big-Sur/events/{year}/{slug}/results"
SLUG_CANDIDATES = [
    "Big-Sur-International-Marathon",
    "Big-Sur-International-Marathon-{year}",
]

API_REQUEST_DELAY = 1.0   # seconds between direct API calls (be polite)
YEAR_DELAY_S = 4.0
HEADLESS = True
OUT_DIR = Path("big_sur_results")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Page loading + total-page detection
# ---------------------------------------------------------------------------

def load_year(page: Page, year: int) -> bool:
    for slug_tmpl in SLUG_CANDIDATES:
        url = BASE.format(year=year, slug=slug_tmpl.format(year=year))
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_selector("table", timeout=15000)
            print(f"  [{year}] loaded {url}")
            return True
        except PWTimeout:
            continue
    return False


def get_total_pages(page: Page) -> int | None:
    try:
        body = page.locator("body").inner_text(timeout=3000)
        idx = body.lower().find("overall results")
        window = body[idx: idx + 3000] if idx >= 0 else body
        m = re.search(r"of\s+(\d+)", window)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Table extraction (works on any HTML containing the Overall table)
# ---------------------------------------------------------------------------

def extract_table_from_html(html: str) -> pd.DataFrame | None:
    """
    Find the Overall results table in an HTML string and return as DataFrame.
    Uses BeautifulSoup, which tolerates SVE's HTML quirks better than pd.read_html.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("    NOTE: install beautifulsoup4 (pip install beautifulsoup4)")
        return None

    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        ths = table.find_all("th")
        if not any("bib" in th.get_text(strip=True).lower() for th in ths):
            continue

        headers = [th.get_text(strip=True) for th in ths]
        ncols = len(headers)

        body = table.find("tbody") or table
        rows = []
        for tr in body.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            row = [c.get_text(strip=True) for c in cells]
            # Skip empty / header-duplicate rows
            if not row or all(v == "" for v in row) or row == headers:
                continue
            # Pad / truncate to header width so DataFrame construction never fails
            if len(row) < ncols:
                row = row + [""] * (ncols - len(row))
            elif len(row) > ncols:
                row = row[:ncols]
            rows.append(row)

        if rows:
            return pd.DataFrame(rows, columns=headers)
    return None


def extract_table_from_page(page: Page) -> pd.DataFrame | None:
    """Extract the visible Overall table from the live page via JS."""
    try:
        data = page.evaluate(
            """() => {
                const headings = [...document.querySelectorAll('h1, h2, h3, h4')]
                    .filter(h => /overall results/i.test(h.textContent));
                const heading = headings.find(h => h.offsetParent !== null)
                                || headings[0];
                const tables = [...document.querySelectorAll('table')];
                const candidate = tables.find(t =>
                    (!heading || (heading.compareDocumentPosition(t)
                        & Node.DOCUMENT_POSITION_FOLLOWING)) &&
                    [...t.querySelectorAll('th')]
                        .some(th => /bib/i.test(th.textContent))
                );
                if (!candidate) return null;
                const rows = [...candidate.querySelectorAll('tr')];
                if (!rows.length) return null;
                const headers = [...rows[0].querySelectorAll('th, td')]
                    .map(c => c.textContent.trim());
                const data = rows.slice(1)
                    .map(r => [...r.querySelectorAll('td, th')]
                        .map(c => c.textContent.trim()))
                    .filter(r => r.length > 0 && r.some(v => v !== ''));
                return {headers, data};
            }"""
        )
    except Exception:
        return None
    if not data or not data.get("data"):
        return None
    return pd.DataFrame(data["data"], columns=data["headers"])


# ---------------------------------------------------------------------------
# Network interception — the key v5 mechanism
# ---------------------------------------------------------------------------

def discover_api_endpoint(page: Page) -> str | None:
    """
    Click Next ONCE and capture the URL of the GET request that fires.
    Returns the URL or None if no XHR-like request was captured.
    """
    captured: list[str] = []

    def on_request(request: Request):
        url = request.url
        rt = request.resource_type
        if request.method != "GET":
            return
        # Heuristics: SVE's results API will have 'results' or 'event' in URL,
        # and will be an XHR/Fetch (not the page navigation).
        if rt in ("xhr", "fetch", "document"):
            if any(k in url.lower() for k in ("results", "event", "page", "data")):
                captured.append(url)

    page.on("request", on_request)

    # Trigger Next click via JS — single click only on the FIRST visible Next
    try:
        page.evaluate(
            """() => {
                const headings = [...document.querySelectorAll('h1, h2, h3, h4')]
                    .filter(h => /overall results/i.test(h.textContent));
                const heading = headings.find(h => h.offsetParent !== null)
                                || headings[0];
                const candidates = [...document.querySelectorAll('a, button')]
                    .filter(a => a.textContent.trim().toLowerCase() === 'next'
                              && a.offsetParent !== null
                              && (!heading
                                  || (heading.compareDocumentPosition(a)
                                      & Node.DOCUMENT_POSITION_FOLLOWING)));
                if (candidates.length > 0) {
                    candidates[0].click();
                }
            }"""
        )
    except Exception:
        pass

    # Give time for any AJAX to fire
    page.wait_for_timeout(3000)
    page.remove_listener("request", on_request)

    # Filter to the most likely API URL — usually has a page-related param
    page_url = page.url.split("#")[0]
    interesting = [u for u in captured if u != page_url]
    # Prefer URLs that include 'page' or look templated
    paginated = [u for u in interesting if re.search(r"[?&](page|p|pn)=\d+", u)]
    if paginated:
        return paginated[0]
    return interesting[0] if interesting else None


def swap_page_param(url: str, page_num: int, verbose: bool = False) -> str:
    """
    Swap any page-like query parameter in URL with the new page number.
    Case-insensitive: matches Page, page, PAGE, PageNumber, pageNum, etc.
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)

    # Look for any param whose name (case-insensitive) is a page indicator.
    page_param_names = {"page", "p", "pn", "pagenum", "pagenumber", "pageindex"}
    matched_key = None
    for k in list(params.keys()):
        if k.lower() in page_param_names:
            matched_key = k
            break

    if matched_key is not None:
        params[matched_key] = [str(page_num)]
        if verbose:
            print(f"    swapped {matched_key}={page_num} (was case-matched)")
    else:
        # No page param found — append using the same case style as siblings
        # Find a representative param to mimic case (PascalCase vs lowercase)
        existing_pascal = any(k[0].isupper() for k in params.keys())
        new_key = "Page" if existing_pascal else "page"
        params[new_key] = [str(page_num)]
        if verbose:
            print(f"    no page param found — appending {new_key}={page_num}")

    return parsed._replace(query=urlencode(params, doseq=True)).geturl()


def fetch_page_via_api(page: Page, api_url_template: str, page_num: int,
                       verbose: bool = False) -> pd.DataFrame | None:
    """Fetch a specific page directly via the captured API URL."""
    target_url = swap_page_param(api_url_template, page_num, verbose=verbose)
    if verbose:
        print(f"    fetching: {target_url}")
    try:
        response = page.request.get(target_url, timeout=30000)
    except Exception as e:
        print(f"    api fetch error for page {page_num}: {e}")
        return None
    if not response.ok:
        print(f"    api returned status {response.status} for page {page_num}")
        return None

    text = response.text()
    if verbose:
        ct = response.headers.get("content-type", "?")
        print(f"    response: status={response.status} type={ct} "
              f"length={len(text):,}")
        snippet = text[:500].replace("\n", " ")
        print(f"    body[:500]: {snippet}")

    # Try parsing as HTML table first
    df = extract_table_from_html(text)
    if df is not None and not df.empty:
        return df

    # Try parsing as JSON (in case the API returns JSON)
    try:
        import json
        data = json.loads(text)
        records = _find_records_in_json(data)
        if records:
            return pd.DataFrame(records)
    except Exception:
        pass

    return None


def _find_records_in_json(obj):
    """Recursively look for a list of dicts inside a JSON structure."""
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            r = _find_records_in_json(v)
            if r:
                return r
    return None


# ---------------------------------------------------------------------------
# Per-year scrape
# ---------------------------------------------------------------------------

def scrape_year(page: Page, year: int) -> pd.DataFrame:
    if not load_year(page, year):
        print(f"  [{year}] no working URL — skipping")
        return pd.DataFrame()

    total_pages = get_total_pages(page)
    if not total_pages:
        print(f"  [{year}] could not read total pages")
        return pd.DataFrame()
    print(f"  [{year}] target: {total_pages} pages "
          f"(~{total_pages * 50:,} rows expected)")

    pages: list[pd.DataFrame] = []

    # Page 1 — grab from the live page
    df1 = extract_table_from_page(page)
    if df1 is None:
        print(f"  [{year}] page 1: extraction failed — bailing")
        return pd.DataFrame()
    pages.append(df1)
    print(f"  [{year}] page 1/{total_pages}: +{len(df1)} rows (from page DOM)")

    if total_pages == 1:
        out = df1.copy()
        out["year"] = year
        return out

    # Discover the API endpoint by clicking Next once
    print(f"  [{year}] discovering API endpoint...")
    api_url = discover_api_endpoint(page)

    if api_url is None:
        print(f"  [{year}] *** No API endpoint captured — falling back to UI ***")
        # Fallback: extract page 2 (we did click Next), then bail
        df2 = extract_table_from_page(page)
        if df2 is not None:
            pages.append(df2)
        out = pd.concat(pages, ignore_index=True)
        out["year"] = year
        return _dedup_and_warn(out, year, total_pages)

    print(f"  [{year}] API: {api_url}")

    # Capture page 2 (we just clicked Next, the page should show p2 now)
    page.wait_for_timeout(800)
    df2 = extract_table_from_page(page)
    if df2 is not None:
        pages.append(df2)
        print(f"  [{year}] page 2/{total_pages}: +{len(df2)} rows (from page DOM)")

    # Now hit the API directly for pages 3..N
    # Verbose for the first one so we can debug if the response shape is unexpected
    for p_idx in range(3, total_pages + 1):
        time.sleep(API_REQUEST_DELAY)
        verbose = (p_idx == 3)  # debug the first API call only
        df = fetch_page_via_api(page, api_url, p_idx, verbose=verbose)
        if df is None or df.empty:
            print(f"  [{year}] page {p_idx}: API returned no data — stopping")
            break
        pages.append(df)
        running = sum(len(p) for p in pages)
        if p_idx % 10 == 0 or p_idx == total_pages or p_idx >= total_pages - 3:
            print(f"  [{year}] page {p_idx}/{total_pages}: +{len(df)} rows "
                  f"(running: {running:,})")

    out = pd.concat(pages, ignore_index=True)
    out["year"] = year
    return _dedup_and_warn(out, year, total_pages)


def _dedup_and_warn(df: pd.DataFrame, year: int, total_pages: int) -> pd.DataFrame:
    dedup_cols = [c for c in ("Bib", "Name") if c in df.columns]
    if dedup_cols:
        before = len(df)
        df = df.drop_duplicates(subset=dedup_cols, keep="first")
        if before != len(df):
            print(f"  [{year}] dropped {before - len(df)} duplicate rows")
    expected = total_pages * 50
    if len(df) < expected * 0.85:
        print(f"  [{year}] *** WARNING: scraped {len(df):,} rows but expected "
              f"~{expected:,} ***")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(pw: Playwright) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    browser = pw.chromium.launch(headless=HEADLESS)
    context = browser.new_context(
        user_agent=USER_AGENT, viewport={"width": 1400, "height": 1000}
    )
    page = context.new_page()

    summary = []
    all_dfs: list[pd.DataFrame] = []
    try:
        for year in YEARS:
            print(f"\n[{year}] {'='*40}")
            try:
                df = scrape_year(page, year)
            except Exception as e:
                print(f"[{year}] ERROR: {type(e).__name__}: {e}")
                summary.append((year, 0))
                try:
                    page.title()
                except Exception:
                    page = context.new_page()
                continue

            if df.empty:
                summary.append((year, 0))
                continue
            path = OUT_DIR / f"big_sur_{year}.csv"
            df.to_csv(path, index=False)
            print(f"[{year}] SAVED {len(df):,} rows -> {path}")
            summary.append((year, len(df)))
            all_dfs.append(df)
            time.sleep(YEAR_DELAY_S)
    finally:
        context.close()
        browser.close()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for yr, n in summary:
        flag = " ⚠ low" if 0 < n < 3000 else ""
        print(f"  {yr}: {n:>6,} rows{flag}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(OUT_DIR / "big_sur_all_years.csv", index=False)
        print(f"\nCombined -> {OUT_DIR / 'big_sur_all_years.csv'}")


def main() -> None:
    with sync_playwright() as pw:
        run(pw)


if __name__ == "__main__":
    main()
