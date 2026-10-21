"""Record an animated GIF tour of the UI with headless Chrome over the DevTools protocol.

    python -m sift.server &                     # in another terminal
    python scripts/record_tour.py --out docs/assets/tour.gif   # needs websocket-client and Pillow

The tour types a query, waits for the cited answer, clicks a citation chip, then visits the
Trace and Map views. Frames are deduplicated and share one palette, which keeps the GIF small
and stops colours flickering between frames.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket  # websocket-client
from PIL import Image

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


class Recorder:
    def __init__(self, ws_url: str, fps: float, max_width: int):
        self.ws = websocket.create_connection(ws_url, timeout=60, suppress_origin=True)
        self.i = 0
        self.fps = fps
        self.max_width = max_width
        self.frames: list[Image.Image] = []
        self.shown_at: list[float] = []

    def call(self, method: str, **params):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.i:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    def js(self, expr: str):
        return self.call("Runtime.evaluate", expression=expr, returnByValue=True).get("result", {}).get("value")

    def wait_for(self, expr: str, timeout: float = 60) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.js(expr):
                return
            self.snap()
            time.sleep(0.05)
        raise TimeoutError(expr)

    def snap(self) -> None:
        shot = self.call("Page.captureScreenshot", format="png")
        img = Image.open(io.BytesIO(base64.b64decode(shot["data"]))).convert("RGB")
        if img.width > self.max_width:
            img = img.resize((self.max_width, round(img.height * self.max_width / img.width)), Image.LANCZOS)
        if not self.frames or img.tobytes() != self.frames[-1].tobytes():
            self.frames.append(img)
            self.shown_at.append(time.time())

    def hold(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            t = time.time()
            self.snap()
            time.sleep(max(0.0, 1 / self.fps - (time.time() - t)))

    def type(self, text: str, delay: float = 0.045) -> None:
        for ch in text:
            self.call("Input.insertText", text=ch)
            time.sleep(delay)
            self.snap()

    def save(self, out: Path, colors: int = 192) -> None:
        durations = [max(20, int((b - a) * 1000)) for a, b in zip(self.shown_at, self.shown_at[1:])] + [3000]
        # build one palette from a strip of frames spread over the tour
        picks = self.frames[:: max(1, len(self.frames) // 6)][:6]
        w, h = picks[0].size
        strip = Image.new("RGB", (w, h * len(picks)))
        for k, f in enumerate(picks):
            strip.paste(f, (0, h * k))
        palette = strip.quantize(colors=colors, method=Image.MEDIANCUT)
        quant = [f.quantize(palette=palette, dither=Image.NONE) for f in self.frames]
        out.parent.mkdir(parents=True, exist_ok=True)
        quant[0].save(out, save_all=True, append_images=quant[1:], duration=durations, loop=0, optimize=True)
        print(f"[tour] {out}: {len(quant)} frames, {out.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8002/")
    ap.add_argument("--query", default="Autophagy declines in aged organisms.")
    ap.add_argument("--out", type=Path, default=Path("docs/assets/tour.gif"))
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--max-width", type=int, default=1200)
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--port", type=int, default=9335)
    args = ap.parse_args()

    found = [c for c in CHROME_CANDIDATES if Path(c).exists()]
    chrome = found[0] if found else shutil.which("chrome") or shutil.which("chromium")
    if not chrome:
        raise SystemExit("Chrome/Chromium not found")
    profile = tempfile.mkdtemp(prefix="sift-tour-")
    flags = ["--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={profile}",
             f"--remote-debugging-port={args.port}", "--remote-allow-origins=*"]
    proc = subprocess.Popen([chrome, *flags, "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/json"))
                break
            except OSError:
                time.sleep(0.1)
        rec = Recorder(next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"], args.fps, args.max_width)
        rec.call("Page.enable")
        rec.call("Runtime.enable")
        rec.call("Emulation.setDeviceMetricsOverride", width=args.width, height=args.height,
                 deviceScaleFactor=1, mobile=False)
        rec.call("Page.navigate", url=f"{args.base}?tour={time.time_ns()}#ask")
        rec.wait_for("document.querySelector('#suggestions .suggestion') !== null")
        rec.hold(1.2)

        rec.js("document.querySelector('#query-input').focus()")
        rec.type(args.query)
        rec.hold(0.4)
        rec.js("document.querySelector('#query-form').requestSubmit()")
        rec.wait_for("document.querySelector('#answer-body .cite') !== null && "
                     "document.querySelector('#source-list li') !== null")
        rec.hold(2.5)
        rec.js("document.querySelector('#answer-body .cite')?.click()")
        rec.hold(2.5)

        rec.js("location.hash = '#trace'")
        rec.hold(3.5)
        rec.js("location.hash = '#map'")
        rec.hold(3.5)
        rec.save(args.out)
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
