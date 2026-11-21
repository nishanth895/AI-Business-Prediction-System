"""capture_screenshots.py

Drives a headless Chromium browser through the Streamlit dashboard and
captures one full-page PNG per page. Saves them under ``screenshots/``.

Requires the Streamlit app to be running on http://127.0.0.1:8555 - start it
with:

    streamlit run src/app.py --server.port 8555 --server.headless true

Run this script from the project root:

    python capture_screenshots.py
"""
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
SHOTS_DIR = ROOT / "screenshots"
SHOTS_DIR.mkdir(parents=True, exist_ok=True)

URL = "http://127.0.0.1:8555/"

PAGES = [
    ("01_overview", "Overview"),
    ("02_business_analysis", "Business Analysis"),
    ("03_sales_forecast", "Sales Forecast"),
    ("04_predictive_alerts", "Predictive Alerts"),
    ("05_business_recommendations", "Business Recommendations"),
]


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 1024},
            device_scale_factor=2,
        )
        page = context.new_page()

        print(f"Loading {URL} ...")
        page.goto(URL, wait_until="networkidle", timeout=60_000)
        # Give Streamlit time to compute caches + render charts
        time.sleep(6)
        # Dismiss the "Run?" / "Wait?" dialogs if any
        for label in ("Rerun", "Continue", "OK"):
            try:
                page.get_by_text(label, exact=True).click(timeout=500)
            except Exception:
                pass

        for filename, label in PAGES:
            print(f"\nSelecting sidebar radio: {label}")
            # Streamlit radios render as label elements in the sidebar
            radio_label = page.locator("label", has_text=label).first
            try:
                radio_label.click(timeout=5_000)
            except Exception as e:
                print(f"  click via label failed ({e}); trying radio input ...")
                page.locator(f"[role='radio']:has-text('{label}')").first.click(timeout=5_000)

            # Wait for the new page content + plotly charts to finish rendering
            time.sleep(5)
            try:
                page.wait_for_load_state("networkidle", timeout=15_000)
            except Exception:
                pass

            out = SHOTS_DIR / f"{filename}.png"
            page.screenshot(path=str(out), full_page=True)
            print(f"  saved -> {out.relative_to(ROOT)}  ({out.stat().st_size:,} bytes)")

        browser.close()
    print("\nAll screenshots captured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
