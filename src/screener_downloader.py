"""
screener_downloader.py
──────────────────────────────────────────────────────────────────────────────
Bulk-downloads the "Export to Excel" file from screener.in for every company
in the NIFTY 500 list (ind_nifty500list.csv).

Usage
─────
  # First run – download all
  python screener_downloader.py

  # Resume a partial run – skip already-downloaded files
  python screener_downloader.py --resume

  # Test a single company (good for checking login / button selectors)
  python screener_downloader.py --test HINDALCO

  # Show help
  python screener_downloader.py --help

Setup
─────
  1. pip install -r requirements.txt
  2. playwright install chromium
  3. Copy .env.example → .env and fill in your Screener.in email + password
  4. Run the script

Output
──────
  screener_exports/          ← one .xlsx per company
  download_log.csv           ← per-company status log
"""

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# ─── Configuration ────────────────────────────────────────────────────────────

load_dotenv()

CSV_FILE      = Path("data/ind_nifty500list.csv") if Path("data/ind_nifty500list.csv").exists() else Path("ind_nifty500list.csv")
LOG_FILE      = Path("logs/download_log.csv") if Path("logs").exists() else Path("download_log.csv")
SCREENER_URL  = "https://www.screener.in"
LOGIN_URL     = f"{SCREENER_URL}/login/"

EMAIL         = os.getenv("SCREENER_EMAIL", "")
PASSWORD      = os.getenv("SCREENER_PASSWORD", "")
DOWNLOAD_DIR  = Path(os.getenv("DOWNLOAD_DIR", "./screener_exports"))
DELAY         = float(os.getenv("DELAY_SECONDS", "3"))
DEFAULT_TYPE  = os.getenv("DATA_TYPE", "consolidated")   # "consolidated" | "standalone"

# ─── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("screener_downloader.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def read_companies(csv_file: Path) -> list[dict]:
    """Read the NIFTY 500 CSV and return a list of dicts with 'symbol' and 'name'."""
    companies = []
    with csv_file.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row.get("Symbol", "").strip()
            name   = row.get("Company Name", "").strip()
            if symbol:                      # skip blank/header rows
                companies.append({"symbol": symbol, "name": name})
    log.info(f"Loaded {len(companies)} companies from {csv_file}")
    return companies


def already_downloaded(symbol: str) -> bool:
    """Return True if the Excel file for this symbol already exists."""
    return (DOWNLOAD_DIR / f"{symbol}.xlsx").exists()


def init_log_file():
    """Create the download log CSV with headers if it doesn't exist."""
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol", "company_name", "status", "data_type", "timestamp", "notes"])


def append_log(symbol: str, name: str, status: str, data_type: str, notes: str = ""):
    """Append a row to the download log."""
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([symbol, name, status, data_type, datetime.now().isoformat(), notes])


# ─── Screener.in interaction ──────────────────────────────────────────────────

def login(page, email: str, password: str) -> bool:
    """Log in to Screener.in. Returns True on success."""
    log.info("Navigating to login page …")
    page.goto(LOGIN_URL, wait_until="domcontentloaded")

    # Fill credentials
    page.fill('input[name="username"]', email)
    page.fill('input[name="password"]', password)
    page.click('button[type="submit"]')

    # Wait for redirect away from login page
    try:
        page.wait_for_url(lambda url: "/login" not in url, timeout=15_000)
        log.info("✅ Login successful")
        return True
    except PlaywrightTimeoutError:
        log.error("❌ Login failed — still on login page. Check your credentials in .env")
        return False


def download_company(page, context, symbol: str, name: str, resume: bool) -> tuple[str, str, str]:
    """
    Download the Excel export for one company.

    Returns (status, data_type_used, notes) where status is one of:
      'success', 'skipped', 'failed', 'not_found'
    """
    if resume and already_downloaded(symbol):
        log.info(f"  ⏩ Skipping {symbol} — already downloaded")
        return "skipped", "-", "file already exists"

    # Try consolidated first, fall back to standalone, then bare URL (no type suffix)
    def _make_urls():
        alt = "standalone" if DEFAULT_TYPE == "consolidated" else "consolidated"
        return [
            (DEFAULT_TYPE, f"{SCREENER_URL}/company/{symbol}/{DEFAULT_TYPE}/"),
            (alt,          f"{SCREENER_URL}/company/{symbol}/{alt}/"),
            ("auto",       f"{SCREENER_URL}/company/{symbol}/"),   # Screener picks best view
        ]

    for data_type, url in _make_urls():
        log.info(f"  → {symbol}  ({data_type})  {url}")

        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        except PlaywrightTimeoutError:
            log.warning(f"    ⚠️  Timeout loading page for {symbol}")
            continue

        # Check for 404 / company-not-found
        if response and response.status == 404:
            log.warning(f"    ⚠️  {symbol} returned 404 on {data_type}")
            continue

        # Check that the page actually has company content (not an error page)
        if page.query_selector("h1.h2") is None and page.query_selector(".company-name") is None:
            # Screener shows "Company not found" text on invalid symbols
            page_text = page.inner_text("body")
            if "not found" in page_text.lower() or "404" in page_text:
                log.warning(f"    ⚠️  {symbol}: company not found on {data_type}")
                continue

        # Look for the Export to Excel button / link
        # Screener uses an <a> tag with "Export to Excel" text or a data attribute
        export_selectors = [
            "a:has-text('Export to Excel')",
            "a[href*='export']",
            "button:has-text('Export to Excel')",
            ".export-button",
        ]

        export_element = None
        for selector in export_selectors:
            try:
                el = page.wait_for_selector(selector, timeout=8_000)
                if el:
                    export_element = el
                    break
            except PlaywrightTimeoutError:
                continue

        if export_element is None:
            log.warning(f"    ⚠️  {symbol}: Export to Excel button not found on {data_type} page")
            continue

        # Set up download handler and click
        try:
            with page.expect_download(timeout=60_000) as download_info:
                export_element.click()
            download = download_info.value

            # Save to our output directory with the symbol as filename
            dest = DOWNLOAD_DIR / f"{symbol}.xlsx"
            download.save_as(dest)

            size_kb = dest.stat().st_size // 1024
            log.info(f"    ✅ Saved {dest.name}  ({size_kb} KB)")
            return "success", data_type, f"{size_kb} KB"

        except PlaywrightTimeoutError:
            log.warning(f"    ⚠️  {symbol}: Download timed out on {data_type}")
            continue
        except Exception as e:
            log.warning(f"    ⚠️  {symbol}: Error during download — {e}")
            continue

    # All attempts failed
    log.error(f"    ❌ {symbol}: Could not download from any data type")
    return "failed", "-", "not found or export unavailable on all data types"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bulk-download Screener.in Excel exports for NIFTY 500 companies"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip companies whose Excel file already exists in the output folder",
    )
    parser.add_argument(
        "--test",
        metavar="SYMBOL",
        help="Download a single company by symbol (e.g. --test HINDALCO) for quick testing",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run Chrome in headless mode (no visible browser window). Default: visible window.",
    )
    args = parser.parse_args()

    # Validate credentials
    if not EMAIL or not PASSWORD:
        log.error("SCREENER_EMAIL and SCREENER_PASSWORD must be set in your .env file")
        log.error("Copy .env.example → .env and fill in your credentials")
        sys.exit(1)

    # Create output directory
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    init_log_file()

    # Load companies
    companies = read_companies(CSV_FILE)

    # If --test mode, filter to just that symbol
    if args.test:
        symbol_upper = args.test.upper()
        companies = [c for c in companies if c["symbol"].upper() == symbol_upper]
        if not companies:
            # Allow testing any symbol even if not in the CSV
            companies = [{"symbol": symbol_upper, "name": "TEST"}]
        log.info(f"Test mode: downloading {symbol_upper} only")

    # ── Summary before starting ──
    total = len(companies)
    log.info("=" * 60)
    log.info(f"Companies to process : {total}")
    log.info(f"Output folder        : {DOWNLOAD_DIR.resolve()}")
    log.info(f"Resume mode          : {'ON' if args.resume else 'OFF'}")
    log.info(f"Headless Chrome      : {'YES' if args.headless else 'NO (visible window)'}")
    log.info(f"Delay between items  : {DELAY}s")
    log.info(f"Preferred data type  : {DEFAULT_TYPE}")
    log.info("=" * 60)

    # ── Start Playwright ──
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=args.headless,
            channel="chrome",          # use system Chrome if available, else Playwright's Chromium
        )
        context = browser.new_context(
            accept_downloads=True,
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()

        # Log in
        if not login(page, EMAIL, PASSWORD):
            browser.close()
            sys.exit(1)

        # ── Process each company ──
        counts = {"success": 0, "skipped": 0, "failed": 0, "not_found": 0}

        for idx, company in enumerate(companies, start=1):
            symbol = company["symbol"]
            name   = company["name"]

            log.info(f"[{idx:>3}/{total}] {symbol}  —  {name}")

            status, data_type, notes = download_company(
                page, context, symbol, name, resume=args.resume
            )

            append_log(symbol, name, status, data_type, notes)
            counts[status] = counts.get(status, 0) + 1

            # Polite delay (skip after last item or in skipped case)
            if idx < total and status != "skipped":
                time.sleep(DELAY)

        browser.close()

    # ── Final summary ──
    log.info("")
    log.info("=" * 60)
    log.info("DOWNLOAD COMPLETE")
    log.info(f"  ✅ Success  : {counts.get('success',  0)}")
    log.info(f"  ⏩ Skipped  : {counts.get('skipped',  0)}")
    log.info(f"  ❌ Failed   : {counts.get('failed',   0)}")
    log.info(f"  🔍 Log file : {LOG_FILE.resolve()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
