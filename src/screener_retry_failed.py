"""
screener_retry_failed.py
──────────────────────────────────────────────────────────────────────────────
Reads download_log.csv, finds all rows with status='failed', and retries them
using the bare URL:
    https://www.screener.in/company/{SYMBOL}/
(letting Screener.in automatically pick consolidated or standalone)

Usage
─────
  python screener_retry_failed.py

  # Run headless (no visible browser window)
  python screener_retry_failed.py --headless
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

LOG_FILE     = Path("logs/download_log.csv") if Path("logs/download_log.csv").exists() else Path("download_log.csv")
DOWNLOAD_DIR = Path(os.getenv("DOWNLOAD_DIR", "./screener_exports"))
DELAY        = float(os.getenv("DELAY_SECONDS", "3"))
SCREENER_URL = "https://www.screener.in"
LOGIN_URL    = f"{SCREENER_URL}/login/"
EMAIL        = os.getenv("SCREENER_EMAIL", "")
PASSWORD     = os.getenv("SCREENER_PASSWORD", "")

# ─── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("screener_retry.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def read_failed_from_log() -> list[dict]:
    """Read download_log.csv and return only failed entries."""
    if not LOG_FILE.exists():
        log.error(f"{LOG_FILE} not found. Run screener_downloader.py first.")
        sys.exit(1)

    failed = []
    with LOG_FILE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "").strip().lower() == "failed":
                failed.append({
                    "symbol": row["symbol"].strip(),
                    "name":   row["company_name"].strip(),
                })
    return failed


def update_log_entry(symbol: str, new_status: str, new_type: str, notes: str):
    """
    Update the status of an existing row in download_log.csv in-place.
    If symbol appears multiple times (e.g. from a previous retry), only the
    LAST (most recent) occurrence is updated.
    """
    rows = []
    with LOG_FILE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(dict(row))

    # Update the LAST matching failed row for this symbol
    last_idx = None
    for i, row in enumerate(rows):
        if row["symbol"].strip() == symbol and row["status"].strip().lower() == "failed":
            last_idx = i

    if last_idx is not None:
        rows[last_idx]["status"]    = new_status
        rows[last_idx]["data_type"] = new_type
        rows[last_idx]["timestamp"] = datetime.now().isoformat()
        rows[last_idx]["notes"]     = notes
    else:
        # Append as a new row if not found (shouldn't happen normally)
        rows.append({
            "symbol":       symbol,
            "company_name": "",
            "status":       new_status,
            "data_type":    new_type,
            "timestamp":    datetime.now().isoformat(),
            "notes":        notes,
        })

    with LOG_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def login(page, email: str, password: str) -> bool:
    """Log in to Screener.in. Returns True on success."""
    log.info("Navigating to login page …")
    page.goto(LOGIN_URL, wait_until="domcontentloaded")
    page.fill('input[name="username"]', email)
    page.fill('input[name="password"]', password)
    page.click('button[type="submit"]')
    try:
        page.wait_for_url(lambda url: "/login" not in url, timeout=15_000)
        log.info("✅ Login successful")
        return True
    except PlaywrightTimeoutError:
        log.error("❌ Login failed — still on login page. Check your credentials in .env")
        return False


def try_download(page, symbol: str, url: str, data_type: str) -> tuple[bool, str]:
    """
    Try to navigate to `url` and click Export to Excel.
    Returns (success, notes).
    """
    log.info(f"    Trying [{data_type}]: {url}")
    try:
        response = page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    except PlaywrightTimeoutError:
        return False, "page timeout"

    if response and response.status == 404:
        return False, f"404 on {data_type}"

    # Check for company-not-found page
    page_text = page.inner_text("body")
    if "not found" in page_text.lower() or "404" in page_text:
        return False, f"company not found ({data_type})"

    # Find the Export to Excel button
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
        return False, f"export button not found ({data_type})"

    # Click and capture download
    try:
        with page.expect_download(timeout=60_000) as dl_info:
            export_element.click()
        download = dl_info.value
        dest = DOWNLOAD_DIR / f"{symbol}.xlsx"
        download.save_as(dest)
        size_kb = dest.stat().st_size // 1024
        log.info(f"    ✅ Saved {dest.name}  ({size_kb} KB)")
        return True, f"{size_kb} KB"
    except PlaywrightTimeoutError:
        return False, f"download timeout ({data_type})"
    except Exception as e:
        return False, f"error: {e} ({data_type})"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Retry failed Screener.in downloads using the bare company URL"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run Chrome headless (no visible window). Default: visible window.",
    )
    args = parser.parse_args()

    if not EMAIL or not PASSWORD:
        log.error("SCREENER_EMAIL and SCREENER_PASSWORD must be set in your .env file")
        sys.exit(1)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    failed = read_failed_from_log()
    total = len(failed)

    if total == 0:
        log.info("🎉 No failed entries found in download_log.csv — nothing to retry!")
        return

    log.info("=" * 60)
    log.info(f"Failed companies to retry : {total}")
    log.info(f"Strategy                  : bare URL → consolidated → standalone")
    log.info(f"Output folder             : {DOWNLOAD_DIR.resolve()}")
    log.info(f"Headless                  : {'YES' if args.headless else 'NO (visible window)'}")
    log.info("=" * 60)

    for company in failed:
        log.info(f"  Symbols to retry: {company['symbol']}")

    log.info("")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=args.headless,
            channel="chrome",
        )
        context = browser.new_context(
            accept_downloads=True,
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()

        if not login(page, EMAIL, PASSWORD):
            browser.close()
            sys.exit(1)

        counts = {"success": 0, "failed": 0}

        for idx, company in enumerate(failed, start=1):
            symbol = company["symbol"]
            name   = company["name"]

            log.info(f"[{idx:>2}/{total}] {symbol}  —  {name}")

            # URL priority: bare → consolidated → standalone
            url_attempts = [
                ("auto",         f"{SCREENER_URL}/company/{symbol}/"),
                ("consolidated", f"{SCREENER_URL}/company/{symbol}/consolidated/"),
                ("standalone",   f"{SCREENER_URL}/company/{symbol}/standalone/"),
            ]

            succeeded = False
            for data_type, url in url_attempts:
                ok, notes = try_download(page, symbol, url, data_type)
                if ok:
                    update_log_entry(symbol, "success", data_type, notes)
                    counts["success"] += 1
                    succeeded = True
                    break

            if not succeeded:
                log.error(f"    ❌ {symbol}: all URL patterns failed")
                update_log_entry(symbol, "failed", "-", "retry failed on all URL patterns")
                counts["failed"] += 1

            if idx < total:
                time.sleep(DELAY)

        browser.close()

    log.info("")
    log.info("=" * 60)
    log.info("RETRY COMPLETE")
    log.info(f"  ✅ Recovered : {counts['success']} / {total}")
    log.info(f"  ❌ Still failed : {counts['failed']} / {total}")
    log.info(f"  🔍 Log updated : {LOG_FILE.resolve()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
