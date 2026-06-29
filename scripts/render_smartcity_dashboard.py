from pathlib import Path
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "lncs_figures" / "smartcity_management_dashboard.html"
OUT = ROOT / "lncs_figures" / "smartcity_management_playwright.png"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 1120}, device_scale_factor=2)
        page.goto(HTML.as_uri(), wait_until="networkidle")
        page.screenshot(path=str(OUT), full_page=True)
        browser.close()
    print(OUT)


if __name__ == "__main__":
    main()
