"""QA for the live board background: layer order, animation running, frame cost, FX toggle, screenshots."""
import os, sys, json
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7871/"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shots")
os.makedirs(OUT, exist_ok=True)
CHROME = os.path.expandvars(r"%LOCALAPPDATA%\ms-playwright\chromium-1234\chrome-win64\chrome.exe")

PROBE = r"""
async () => {
  const c = document.querySelector('.board-electric');
  const img = document.querySelector('.board-img');
  const lit = () => { const x = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0;
    for (let i = 3; i < x.length; i += 16) if (x[i] > 20) n++; return n; };
  // frame timing over 3 s
  const times = []; let t0 = performance.now(), prev = t0;
  await new Promise(res => { const f = (now) => { times.push(now - prev); prev = now;
    if (now - t0 < 3000) requestAnimationFrame(f); else res(); }; requestAnimationFrame(f); });
  times.sort((a, b) => a - b);
  // cost of our own frame: time spent inside a frame callback is what matters; approximate with long tasks
  const bg = document.querySelector('.board-bg');
  return {
    hasBg: !!bg, imgLoaded: img && img.complete && img.naturalWidth, canvas: [c.width, c.height],
    bgZ: getComputedStyle(bg).zIndex, appZ: getComputedStyle(document.querySelector('gradio-app')).zIndex,
    containerBg: getComputedStyle(document.querySelector('.gradio-container')).backgroundColor,
    tabBg: getComputedStyle([...document.querySelectorAll('.tabitem')].find(t => t.offsetParent)).backgroundColor,
    frames: times.length, fps: Math.round(times.length / 3), p50: +times[Math.floor(times.length / 2)].toFixed(1),
    p95: +times[Math.floor(times.length * .95)].toFixed(1), litPixels: lit(),
    fxBtn: document.querySelector('.ov-fx') && document.querySelector('.ov-fx').textContent,
  };
}
"""

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME) if os.path.exists(CHROME) else p.chromium.launch()
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL, wait_until="networkidle"); pg.wait_for_timeout(3000)
    print("NORMAL", json.dumps(pg.evaluate(PROBE)))
    pg.screenshot(path=os.path.join(OUT, "board_top.png"))
    pg.mouse.wheel(0, 900); pg.wait_for_timeout(700)
    pg.screenshot(path=os.path.join(OUT, "board_mid.png"))
    pg.mouse.wheel(0, -900); pg.wait_for_timeout(300)

    # 4x CPU throttle: does it stay smooth (watchdog)?
    cdp = pg.context.new_cdp_session(pg)
    cdp.send("Emulation.setCPUThrottlingRate", {"rate": 4})
    pg.wait_for_timeout(3500)
    print("THROTTLE4X", json.dumps(pg.evaluate(PROBE)))
    cdp.send("Emulation.setCPUThrottlingRate", {"rate": 1})

    # Chrome's own per-frame script cost, from a performance trace-lite: run 120 frames and time our callback
    cost = pg.evaluate("""async () => {
      const c = document.querySelector('.board-electric'); let busy = 0, n = 0;
      const orig = window.requestAnimationFrame;
      await new Promise(res => { const f = () => { const s = performance.now(); n++; busy += performance.now() - s;
        if (n < 120) orig(f); else res(); }; orig(f); });
      return n; }""")

    # FX toggle off/on
    pg.locator(".ov-fx").click(); pg.wait_for_timeout(800)
    off = pg.evaluate("""() => { const c = document.querySelector('.board-electric');
      const x = c.getContext('2d').getImageData(0,0,c.width,c.height).data; let n=0;
      for (let i=3;i<x.length;i+=16) if (x[i]>20) n++;
      return {btn: document.querySelector('.ov-fx').textContent, lit: n, stored: localStorage.getItem('ov-fx'),
              cls: document.documentElement.classList.contains('ov-fx-off')}; }""")
    print("FX_OFF", off)
    pg.screenshot(path=os.path.join(OUT, "board_off.png"))
    pg.locator(".ov-fx").click(); pg.wait_for_timeout(1500)
    print("FX_ON_AGAIN", pg.evaluate("() => [document.querySelector('.ov-fx').textContent, localStorage.getItem('ov-fx')]"))

    # reduced motion: static board, no animation
    ctx2 = b.new_context(viewport={"width": 1920, "height": 1080}, reduced_motion="reduce")
    pg2 = ctx2.new_page(); pg2.goto(URL, wait_until="networkidle"); pg2.wait_for_timeout(2500)
    print("REDUCED", pg2.evaluate("""() => { const c = document.querySelector('.board-electric');
      const x = c.getContext('2d').getImageData(0,0,c.width,c.height).data; let n=0;
      for (let i=3;i<x.length;i+=16) if (x[i]>20) n++; return {lit: n, img: !!document.querySelector('.board-img')}; }"""))
    print("ERRORS", errs[:8])
    b.close()
