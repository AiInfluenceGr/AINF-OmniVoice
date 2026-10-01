"""README screenshots. Run tools/demo_server.py first (neutral demo clip on :7872), then this with a Playwright Python."""
import os
from playwright.sync_api import sync_playwright
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs")
os.makedirs(OUT, exist_ok=True)
CHROME = os.path.expandvars(r"%LOCALAPPDATA%\ms-playwright\chromium-1234\chrome-win64\chrome.exe")
URL = os.environ.get("SHOT_URL", "http://127.0.0.1:7872/?__theme=dark")
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME)
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto(URL, wait_until="networkidle"); pg.wait_for_timeout(3000)
    panel = pg.locator(".tabitem:visible").first
    panel.locator("textarea").first.fill(
        "This voice was cloned from an eight second clip. It can read anything you type here, "
        "in more than six hundred languages.")
    panel.get_by_label("Extra style (optional)").fill("low pitch")
    pg.wait_for_timeout(2500)
    pg.screenshot(path=os.path.join(OUT, "clone.png"), clip={"x": 0, "y": 0, "width": 1920, "height": 1000})

    pg.get_by_role("tab", name="05 · History").click(); pg.wait_for_timeout(1500)
    pg.wait_for_timeout(2500)
    pg.screenshot(path=os.path.join(OUT, "history.png"), clip={"x": 0, "y": 0, "width": 1920, "height": 1000})
    b.close()
print("ok")
