"""Capture animation frames + an alignment overlay (all traces drawn by the page's own transform) at 1920x1080."""
import os
from playwright.sync_api import sync_playwright
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shots")
CHROME = os.path.expandvars(r"%LOCALAPPDATA%\ms-playwright\chromium-1234\chrome-win64\chrome.exe")
URL = "http://127.0.0.1:7871/?__theme=dark"
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME)
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto(URL, wait_until="networkidle"); pg.wait_for_timeout(3000)
    # hide the UI to see the board alone
    pg.add_style_tag(content="gradio-app{visibility:hidden !important}")
    for i in range(3):
        pg.wait_for_timeout(700)
        pg.screenshot(path=os.path.join(OUT, f"pulse_{i}.png"))
    # zoomed crop of the pulses, 2x device scale
    pg2 = b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    pg2.goto(URL, wait_until="networkidle"); pg2.wait_for_timeout(3500)
    pg2.screenshot(path=os.path.join(OUT, "pulse_ui.png"))
    b.close()
