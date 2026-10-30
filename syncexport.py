"""
Synced-video export for videosync.py
------------------------------------
Renders one MP4 that stacks the Vevo video, a readout line and the
absorbance / intensity traces with a red dotted cursor that moves with the
video, the same view as the videosync.py player.

Used by the "Export synced video" button in videosync.py, and also runnable on
its own (handy for long or large recordings):

    python syncexport.py VIDEO REC_abs.csv REC_counts.csv -o out.mp4 \
        [--wavelength 600] [--halfwidth 2] [--fps 30] [--width 1280] \
        [--start 0] [--end 105] [--no-points]

Timing: video t = 0 s is OD Elapsed_s = 0, as in videosync.py.
"""
import argparse
import os
import re
import subprocess
import warnings

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

META_COLS = ["Clock_time", "Elapsed_s", "IntTime", "NumAvg"]
RED = np.array([214, 39, 40], np.uint8)


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def probe(path: str):
    """(width, height, duration s) of a video file."""
    err = subprocess.run([ffmpeg_exe(), "-i", path], capture_output=True, text=True).stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", err)
    dur = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else float("nan")
    s = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", err)
    if not s:
        raise RuntimeError("could not read video size")
    return int(s[1]), int(s[2]), dur


# ----------------------------------------------------------------------------
# Static plot, rendered once
# ----------------------------------------------------------------------------
def render_plot(t, series, width, show_pts, panel_h=290):
    """series: list of (key, label, y). Returns (rgb array, per-panel pixel mappers, (top, bottom) rows)."""
    dpi = 100
    n = len(series)
    h = panel_h * n + 70
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.weight": "bold", "axes.labelweight": "bold"})
    fig, axes = plt.subplots(n, 1, figsize=(width / dpi, h / dpi), dpi=dpi, sharex=True)
    axes = np.atleast_1d(axes)
    for a, (key, label, y) in zip(axes, series):
        a.plot(t, y, color="k", lw=1.6)
        if show_pts:
            a.plot(t, y, "o", color="k", ms=3)
        a.set_ylabel(label.replace(", ", "\n").replace(" (", "\n("), fontsize=13)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_linewidth(3)
        a.tick_params(direction="out", width=2.5, length=6, labelsize=12)
        for lab in a.get_xticklabels() + a.get_yticklabels():
            lab.set_fontweight("bold")
    axes[-1].set_xlabel("Experimental time (s)", fontsize=14)
    fig.tight_layout(pad=0.8, h_pad=1.2)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    H = rgb.shape[0]

    def mapper(a):
        tr = a.transData

        def f(x, y):
            px, py = tr.transform((x, y))
            return int(round(px)), int(round(H - py))
        return f

    maps = [mapper(a) for a in axes]
    y_top = int(round(H - axes[0].get_window_extent().y1))
    y_bot = int(round(H - axes[-1].get_window_extent().y0))
    plt.close(fig)
    return rgb, maps, (y_top, y_bot)


def _disc(radius, ring):
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    r = np.hypot(xx, yy)
    return r <= radius - ring, (r <= radius) & (r > radius - ring)


DOT_FILL, DOT_RING = _disc(7, 2)


def draw_dot(img, x, y):
    R = DOT_FILL.shape[0] // 2
    H, W = img.shape[:2]
    y0, y1, x0, x1 = y - R, y + R + 1, x - R, x + R + 1
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        return
    sub = img[y0:y1, x0:x1]
    sub[DOT_FILL] = RED
    sub[DOT_RING] = 0


def draw_cursor(img, x, top, bot, width=3, dash=7, gap=5):
    W = img.shape[1]
    xa, xb = max(0, x - width // 2), min(W, x - width // 2 + width)
    if xb <= xa:
        return
    for r in range(top, bot, dash + gap):
        img[r:min(bot, r + dash), xa:xb] = RED


# ----------------------------------------------------------------------------
# Main render
# ----------------------------------------------------------------------------
def render_synced_video(video_path, out_path, t, series, *, fps=30.0, width=1280,
                        start=0.0, end=None, show_pts=True, progress=None):
    """
    video_path : any ffmpeg-readable video (t = 0 is Elapsed_s = 0)
    t          : OD Elapsed_s array
    series     : list of (key, label, y) with key 'abs' or 'counts'
    progress   : optional callback(fraction 0-1)
    """
    t = np.asarray(t, float)
    vw, vh, vdur = probe(video_path)
    if end is None or not np.isfinite(end):
        end = vdur
    end = min(end, vdur) if np.isfinite(vdur) else end
    if end <= start:
        raise ValueError("end time must be after start time")
    width = int(width) // 2 * 2
    v_h = int(round(vh * width / vw)) // 2 * 2

    plot, maps, (ptop, pbot) = render_plot(t, series, width, show_pts)
    bar_h = 44
    H = v_h + bar_h + plot.shape[0]
    H += H % 2
    frame = np.full((H, width, 3), 255, np.uint8)

    font_path = font_manager.findfont(font_manager.FontProperties(
        family=["Arial", "Liberation Sans", "DejaVu Sans"], weight="bold"))
    font = ImageFont.truetype(font_path, 20)
    ys = [np.asarray(y, float) for _, _, y in series]
    keys = [k for k, _, _ in series]

    ff = ffmpeg_exe()
    dec = subprocess.Popen(
        [ff, "-loglevel", "error", "-ss", f"{start:.3f}", "-i", video_path, "-t", f"{end - start:.3f}",
         "-vf", f"fps={fps},scale={width}:{v_h}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = subprocess.Popen(
        [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{H}", "-r", f"{fps}", "-i", "-", "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    n_est = max(1, int(round((end - start) * fps)))
    fsize = width * v_h * 3
    py0 = v_h + bar_h
    i = 0
    try:
        while True:
            buf = dec.stdout.read(fsize)
            if len(buf) < fsize:
                break
            tt = start + i / fps
            k = int(np.argmin(np.abs(t - tt)))
            frame[:v_h] = np.frombuffer(buf, np.uint8).reshape(v_h, width, 3)
            # readout bar
            vals = []
            for key, y in zip(keys, ys):
                v = y[k]
                s = "NaN" if not np.isfinite(v) else (f"{v:.4f}" if key == "abs" else f"{v:.1f}")
                vals.append(("A" if key == "abs" else "I") + " = " + s)
            txt = (f"video t = {tt:.3f} s   |   OD frame {k + 1}/{len(t)} (t = {t[k]:.3f} s)   |   "
                   + "    ".join(vals))
            bar = Image.new("RGB", (width, bar_h), "white")
            ImageDraw.Draw(bar).text((width // 2, bar_h // 2), txt, fill="black", font=font, anchor="mm")
            frame[v_h:py0] = np.asarray(bar)
            # plot + cursor + current dots
            region = frame[py0:py0 + plot.shape[0]]
            region[:] = plot
            cx, _ = maps[0](tt, 0)
            draw_cursor(region, cx, ptop, pbot)
            for m, y in zip(maps, ys):
                if np.isfinite(y[k]):
                    draw_dot(region, *m(t[k], y[k]))
            enc.stdin.write(frame.tobytes())
            i += 1
            if progress and i % 10 == 0:
                progress(min(1.0, i / n_est))
    finally:
        dec.stdout.close()
        dec.wait()
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="ignore")
        enc.wait()
    if enc.returncode != 0:
        raise RuntimeError(err.strip()[-600:] or "ffmpeg encode failed")
    if i == 0:
        raise RuntimeError("no video frames were decoded in that time window")
    if progress:
        progress(1.0)
    return i


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _load(path):
    import pandas as pd
    df = pd.read_csv(path)
    wl_cols = [c for c in df.columns if c not in META_COLS]
    wl = np.array([float(c) for c in wl_cols])
    return df["Elapsed_s"].to_numpy(float), wl, df[wl_cols].to_numpy(np.float32)


def _band(wl, V, target, half):
    sel = np.abs(wl - target) <= half
    if half <= 0 or not sel.any():
        return V[:, int(np.argmin(np.abs(wl - target)))].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(V[:, sel], axis=1).astype(float)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("csvs", nargs="+", help="_abs.csv and/or _counts.csv")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--wavelength", type=float, default=600.0)
    p.add_argument("--halfwidth", type=float, default=2.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--no-points", action="store_true")
    a = p.parse_args()

    tag = f"{a.wavelength:g} ± {a.halfwidth:g} nm" if a.halfwidth else f"{a.wavelength:g} nm"
    found = {}
    for c in a.csvs:
        low = c.lower()
        k = "abs" if low.endswith("_abs.csv") else "counts" if low.endswith("_counts.csv") else None
        if k:
            found[k] = _load(c)
    if not found:
        p.error("pass the _abs.csv and/or _counts.csv file")
    t = found.get("abs", found.get("counts"))[0]
    series = []
    for k, lab in (("abs", f"Absorbance ({tag})"), ("counts", f"Intensity, counts ({tag})")):
        if k in found:
            _, wl, V = found[k]
            series.append((k, lab, _band(wl, V, a.wavelength, a.halfwidth)))
    out = a.out or os.path.splitext(a.video)[0] + f"_synced_{a.wavelength:g}nm.mp4"

    def prog(f):
        print(f"\r{f * 100:5.1f}%", end="", flush=True)

    n = render_synced_video(a.video, out, t, series, fps=a.fps, width=a.width, start=a.start,
                            end=a.end, show_pts=not a.no_points, progress=prog)
    print(f"\nwrote {out} ({n} frames)")


if __name__ == "__main__":
    main()
