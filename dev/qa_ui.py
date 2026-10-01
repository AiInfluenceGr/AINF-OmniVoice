"""QA: measure tab geometry at 1920x1080, check dark theme + fonts, screenshot each tab into shots/."""
import os, sys, json
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7871/"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shots")
os.makedirs(OUT, exist_ok=True)
CHROME = os.path.expandvars(r"%LOCALAPPDATA%\ms-playwright\chromium-1234\chrome-win64\chrome.exe")

JS = r"""
() => {
  const box = (el) => { if (!el) return null; const r = el.getBoundingClientRect();
    return [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]; };
  const vis = [...document.querySelectorAll('.tabitem')].filter(t => getComputedStyle(t).display !== 'none')[0];
  return {container: box(document.querySelector('.gradio-container')), tab: box(vis),
          runbar: box(document.querySelector('.runbar')), console: box(document.querySelector('#ov-console'))};
}
"""
STYLE = r"""
() => {
  const g = (s) => document.querySelector(s);
  const cs = (s, p) => { const e = g(s); return e ? getComputedStyle(e)[p] : null; };
  const cb = [...document.querySelectorAll('input[type=checkbox]')];
  const on = cb.find(c => c.checked), off = cb.find(c => !c.checked);
  return {
    dark: document.body.classList.contains('dark') || location.search.includes('__theme=dark'),
    bodyBg: getComputedStyle(document.body).backgroundColor,
    containerBg: cs('.gradio-container', 'backgroundColor'),
    titleFont: cs('.ov-title', 'fontFamily'), titleWeight: cs('.ov-title', 'fontWeight'),
    tabFont: cs('[role=tab]', 'fontFamily'),
    primaryBg: cs('button.primary', 'backgroundColor'),
    tabSelColor: cs('[role=tab].selected', 'color'),
    cbOn: on ? getComputedStyle(on).backgroundColor : null, cbOff: off ? getComputedStyle(off).backgroundColor : null,
    fontsOk: [...document.fonts].filter(f => f.status === 'loaded').map(f => f.family + ' ' + f.weight),
    errors: document.querySelectorAll('.toast-body.error, .error').length,
    maxRadius: Math.max(...[...document.querySelectorAll('button, textarea, input[type=text], .tabitem')].map(e => parseFloat(getComputedStyle(e).borderTopLeftRadius) || 0)),
  };
}
"""

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME) if os.path.exists(CHROME) else p.chromium.launch()
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.goto(URL, wait_until="networkidle")
    pg.wait_for_timeout(3000)
    print("URL", pg.url)
    print("STYLE", json.dumps(pg.evaluate(STYLE), indent=1))
    tabs = pg.locator("[role=tab]")
    n = tabs.count()
    refs, ok = {}, True
    for i in range(n):
        tabs.nth(i).click()
        pg.wait_for_timeout(900)
        d = pg.evaluate(JS)
        name = tabs.nth(i).inner_text().strip()
        print(f"TAB{i} {name!r}", d)
        for k, v in d.items():
            if v is None or k in ("runbar", "console"):
                # runbar/console y depends on tab min-height; compare x/w only
                if v is not None:
                    v = (v[0], v[2])
                else:
                    continue
            else:
                v = (v[0], v[2], v[3]) if k == "tab" else (v[0], v[2])
            if refs.setdefault(k, v) != v:
                ok = False
                print("   MISMATCH", k, refs[k], v)
        pg.screenshot(path=os.path.join(OUT, f"tab{i}.png"), full_page=True)
    # open the settings accordion and shoot once
    tabs.nth(0).click(); pg.wait_for_timeout(500)
    pg.locator("#ov-settings button").first.click()
    pg.wait_for_timeout(800)
    pg.screenshot(path=os.path.join(OUT, "settings_open.png"), full_page=True)
    print("CONSOLE_ERRORS", errs[:10])
    print("ALL-TABS-IDENTICAL" if ok else "MISMATCH FOUND")
    b.close()
