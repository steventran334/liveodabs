"""
Vevo video + OD sync viewer
---------------------------
Plays a Vevo 2100 video next to the absorbance and intensity time traces from
one spectro.m recording (<name>_abs.csv / <name>_counts.csv). A vertical cursor
through both traces follows the video while it plays or is scrubbed; clicking
the traces seeks the video to that time.

Timing assumption: video t = 0 s is the same instant as Elapsed_s = 0 in the
OD recording (the video was trimmed so both experiments start together).

Any video format ffmpeg can read (.mp4, .avi, .mov, .gif, ...) is converted
once to browser-playable H.264 MP4. GIF frame timing is preserved.

Run:  streamlit run videosync.py
"""
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

META_COLS = ["Clock_time", "Elapsed_s", "IntTime", "NumAvg"]

st.set_page_config(page_title="Vevo + OD sync", layout="wide")
st.title("Vevo video + OD sync")
st.caption(
    "Upload the Vevo 2100 video and the `_abs.csv` / `_counts.csv` files from the "
    "spectro.m recording. Video t = 0 is taken as OD Elapsed_s = 0."
)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading OD file...")
def read_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


def parse_frames(df: pd.DataFrame):
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing expected column(s): {', '.join(missing)}")
    wl_cols = [c for c in df.columns if c not in META_COLS]
    wl = np.array([float(c) for c in wl_cols])
    vals = df[wl_cols].to_numpy(dtype=np.float32)
    return df["Elapsed_s"].to_numpy(float), df["Clock_time"].astype(str).to_numpy(), wl, vals


def classify(name: str, df: pd.DataFrame) -> str:
    if "Wavelength_nm" in df.columns:
        return "meta"
    low = name.lower()
    if low.endswith("_abs.csv"):
        return "abs"
    if low.endswith("_counts.csv"):
        return "counts"
    vals = df.drop(columns=[c for c in META_COLS if c in df.columns]).to_numpy(float)
    return "counts" if np.nanmedian(np.abs(vals)) > 50 else "abs"


def band_trace(wl, V, target, half_width):
    """Mean over target +/- half_width nm (nearest pixel if half_width == 0)."""
    sel = np.abs(wl - target) <= half_width
    if half_width <= 0 or not sel.any():
        i = int(np.argmin(np.abs(wl - target)))
        return V[:, i].astype(float), f"pixel at {wl[i]:.3f} nm"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        y = np.nanmean(V[:, sel], axis=1).astype(float)
    return y, f"mean of {int(sel.sum())} pixels ({wl[sel].min():.2f}-{wl[sel].max():.2f} nm)"


def stem_of(name: str) -> str:
    s = re.sub(r"\.csv$", "", name, flags=re.I)
    return re.sub(r"_(abs|counts|meta)$", "", s, flags=re.I)


@st.cache_data(show_spinner="Converting video to browser-playable MP4...")
def to_browser_mp4(raw: bytes, suffix: str) -> tuple[bytes, float]:
    """Transcode any video to H.264/yuv420p MP4. Returns (mp4 bytes, duration s)."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in" + suffix)
        dst = os.path.join(td, "out.mp4")
        with open(src, "wb") as f:
            f.write(raw)
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", src, "-an",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-pix_fmt", "yuv420p",
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
               "-movflags", "+faststart", dst]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[-600:] or "ffmpeg failed")
        probe = subprocess.run([ffmpeg, "-i", dst], capture_output=True, text=True).stderr
        m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", probe)
        dur = (int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])) if m else float("nan")
        with open(dst, "rb") as f:
            return f.read(), dur


# ----------------------------------------------------------------------------
# Uploads
# ----------------------------------------------------------------------------
with st.sidebar:
    vid_up = st.file_uploader("Vevo video", type=["mp4", "avi", "mov", "gif", "webm", "mkv"])
    csv_ups = st.file_uploader("OD recording (_abs.csv, _counts.csv)", type=["csv"],
                               accept_multiple_files=True)

if not vid_up or not csv_ups:
    st.info("Upload the Vevo video and at least one of `_abs.csv` / `_counts.csv`.")
    st.stop()

files = {}
for u in csv_ups:
    df = read_csv(u.getvalue())
    kind = classify(u.name, df)
    if kind == "meta":
        continue  # not needed here
    if kind in files:
        st.error(f"Two {kind} files uploaded. Upload one recording at a time.")
        st.stop()
    files[kind] = (u.name, df)

if not files:
    st.error("Upload the `_abs.csv` and/or `_counts.csv` file (the `_meta.csv` isn't used here).")
    st.stop()

data = {k: parse_frames(df) for k, (_, df) in files.items()}
first = "abs" if "abs" in data else "counts"
t, clock, wl, _ = data[first]
stem = stem_of(files[first][0])
if len(data) == 2 and not np.allclose(data["abs"][0], data["counts"][0]):
    st.error("The _abs and _counts files come from different recordings.")
    st.stop()

# ----------------------------------------------------------------------------
# Controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Trace")
    target = st.number_input("Wavelength (nm)", min_value=float(np.ceil(wl.min())),
                             max_value=float(np.floor(wl.max())), value=600.0, step=1.0)
    half = st.number_input("Average over ± nm", min_value=0.0, max_value=50.0,
                           value=2.0, step=0.5,
                           help="0 = single nearest pixel. Averaging a few nm cuts pixel noise.")
    show_pts = st.checkbox("Show individual OD points", value=True)
    st.header("Playback")
    rate = st.select_slider("Speed", options=[0.25, 0.5, 1.0, 2.0, 4.0], value=1.0)
    height = st.slider("Video height (px)", 200, 800, 380, step=20)

traces, notes = {}, []
labels = {"abs": f"Absorbance ({target:g} ± {half:g} nm)" if half else f"Absorbance ({target:g} nm)",
          "counts": f"Intensity, counts ({target:g} ± {half:g} nm)" if half else f"Intensity, counts ({target:g} nm)"}
for k in ("abs", "counts"):
    if k in data:
        y, note = band_trace(wl, data[k][3], target, half)
        traces[k] = y
        notes.append(f"{'A' if k == 'abs' else 'I'}: {note}")

try:
    mp4, vdur = to_browser_mp4(vid_up.getvalue(), os.path.splitext(vid_up.name)[1].lower())
except Exception as e:
    st.error(f"Could not read the video: {e}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Video length", f"{vdur:.2f} s")
c2.metric("OD length", f"{t[-1]:.2f} s")
c3.metric("OD frames", f"{len(t)}")
c4.metric("OD started", clock[0][11:] if len(clock[0]) > 11 else clock[0])
if np.isfinite(vdur) and abs(vdur - t[-1]) > max(1.0, 0.05 * t[-1]):
    st.warning(f"Video ({vdur:.1f} s) and OD recording ({t[-1]:.1f} s) differ in length. "
               "The cursor follows video time, so outside the overlap it sits past the OD data. "
               "If the starts don't line up, re-trim the video (e.g. in vevo2100-extract).")
if len(mp4) > 150e6:
    st.warning("The converted video is large and may load slowly. Trim or downscale it if playback stutters.")

# ----------------------------------------------------------------------------
# Synced player (HTML5 video + Plotly, synced in the browser)
# ----------------------------------------------------------------------------
payload = {
    "t": [round(float(x), 4) for x in t],
    "clock": [str(c) for c in clock],
    "series": [{"key": k, "label": labels[k],
                "y": [None if not np.isfinite(v) else round(float(v), 6) for v in traces[k]]}
               for k in ("abs", "counts") if k in traces],
    "rate": rate,
    "markers": show_pts,
    "vdur": vdur if np.isfinite(vdur) else None,
}
vid_b64 = base64.b64encode(mp4).decode()


@st.cache_resource
def plotly_script() -> str:
    """Inline plotly.js from the Python package (works offline); CDN as fallback."""
    try:
        from plotly.offline import get_plotlyjs
        return "<script>" + get_plotlyjs() + "</script>"
    except Exception:
        return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'


n_panels = len(payload["series"])
plot_h = 230 * n_panels + 60

html = plotly_script() + f"""
<style>
  body {{ margin:0; font-family: Arial, Helvetica, sans-serif; }}
  #wrap {{ display:flex; flex-direction:column; gap:6px; }}
  video {{ height:{height}px; max-width:100%; background:#000; align-self:center; }}
  #bar {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; font-size:14px; }}
  button {{ font: 600 13px Arial; padding:4px 10px; border:1.5px solid #000; background:#fff;
            border-radius:4px; cursor:pointer; }}
  button:hover {{ background:#eee; }}
  #readout {{ font-weight:700; font-variant-numeric: tabular-nums; }}
</style>
<div id="wrap">
  <video id="v" controls preload="auto" playsinline src="data:video/mp4;base64,{vid_b64}"></video>
  <div id="bar">
    <button id="prev" title="Previous OD frame (←)">◀ OD frame</button>
    <button id="play">Play / pause</button>
    <button id="next" title="Next OD frame (→)">OD frame ▶</button>
    <span id="readout"></span>
  </div>
  <div id="plot" style="height:{plot_h}px;"></div>
</div>
<script>
const D = {json.dumps(payload)};
const v = document.getElementById('v');
v.playbackRate = D.rate;
const n = D.series.length;
const traces = [], layout = {{
  margin: {{l:80, r:20, t:10, b:50}}, showlegend:false, hovermode:'closest',
  font: {{family:'Arial', size:13, color:'#000'}}, plot_bgcolor:'#fff', paper_bgcolor:'#fff',
  shapes: [], grid: {{rows:n, columns:1, pattern:'independent', roworder:'top to bottom'}}
}};
const ax = (extra) => Object.assign({{showline:true, linewidth:2.5, linecolor:'#000', ticks:'outside',
  tickwidth:2, ticklen:6, mirror:false, showgrid:false, zeroline:false,
  tickfont:{{family:'Arial Black, Arial', size:12}}}}, extra);
D.series.forEach((s, i) => {{
  const sfx = i === 0 ? '' : String(i + 1);
  traces.push({{x:D.t, y:s.y, xaxis:'x'+sfx, yaxis:'y'+sfx, type:'scatter',
    mode: D.markers ? 'lines+markers' : 'lines',
    line:{{color:'#000', width:1.5}}, marker:{{color:'#000', size:3.5}},
    hovertemplate:'t = %{{x:.3f}} s<br>%{{y:.4g}}<extra></extra>'}});
  // current-value dot
  traces.push({{x:[D.t[0]], y:[s.y[0]], xaxis:'x'+sfx, yaxis:'y'+sfx, type:'scatter', mode:'markers',
    marker:{{color:'#d62728', size:10, line:{{color:'#000', width:1.5}}}}, hoverinfo:'skip'}});
  layout['xaxis'+sfx] = ax({{matches: i ? 'x' : undefined, showticklabels: i === n-1,
    title: i === n-1 ? {{text:'<b>Experimental time (s)</b>'}} : undefined}});
  layout['yaxis'+sfx] = ax({{title:{{text:'<b>'+s.label.replace(', ', '<br>')+'</b>', font:{{size:13}}}}}});
}});
layout.shapes.push({{type:'line', xref:'x', yref:'paper', x0:0, x1:0, y0:0, y1:1,
  line:{{color:'#d62728', width:2, dash:'dot'}}}});
const plot = document.getElementById('plot');
Plotly.newPlot(plot, traces, layout, {{displaylogo:false, responsive:true,
  modeBarButtonsToRemove:['lasso2d','select2d']}});

function nearest(tt) {{
  let lo = 0, hi = D.t.length - 1;
  while (hi - lo > 1) {{ const m = (lo + hi) >> 1; if (D.t[m] <= tt) lo = m; else hi = m; }}
  return (tt - D.t[lo] <= D.t[hi] - tt) ? lo : hi;
}}
const ro = document.getElementById('readout');
let lastK = -1, lastT = -1;
function update() {{
  const tt = v.currentTime;
  if (tt === lastT) return; lastT = tt;
  const k = nearest(tt);
  const upd = {{'shapes[0].x0': tt, 'shapes[0].x1': tt}};
  if (k !== lastK) {{
    lastK = k;
    const xs = [], ys = [], idx = [];
    D.series.forEach((s, i) => {{ xs.push([D.t[k]]); ys.push([s.y[k]]); idx.push(2*i + 1); }});
    Plotly.restyle(plot, {{x: xs, y: ys}}, idx);
  }}
  Plotly.relayout(plot, upd);
  const vals = D.series.map(s => (s.key === 'abs' ? 'A' : 'I') + ' = ' +
     (s.y[k] === null ? 'NaN' : (s.key === 'abs' ? s.y[k].toFixed(4) : s.y[k].toFixed(1))));
  const out = (tt > D.t[D.t.length-1] + 0.5) ? '  (past end of OD data)' : '';
  ro.textContent = 'video t = ' + tt.toFixed(3) + ' s  |  OD frame ' + (k+1) + '/' + D.t.length +
     ' (t = ' + D.t[k].toFixed(3) + ' s)  |  ' + vals.join('   ') + out;
}}
function loop() {{ update(); if (!v.paused && !v.ended) requestAnimationFrame(loop); }}
v.addEventListener('play', () => requestAnimationFrame(loop));
['seeked','seeking','timeupdate','loadedmetadata'].forEach(e => v.addEventListener(e, update));

function seek(tt) {{ v.pause(); v.currentTime = Math.max(0, Math.min(tt, v.duration || tt)); }}
plot.on('plotly_click', ev => {{ if (ev.points.length) seek(ev.points[0].x); }});
document.getElementById('play').onclick = () => v.paused ? v.play() : v.pause();
const step = d => {{ const k = nearest(v.currentTime);
  let j = k + d; if (d > 0 && D.t[k] > v.currentTime + 1e-3) j = k;
  if (d < 0 && D.t[k] < v.currentTime - 1e-3) j = k;
  seek(D.t[Math.max(0, Math.min(D.t.length-1, j))]); }};
document.getElementById('prev').onclick = () => step(-1);
document.getElementById('next').onclick = () => step(1);
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft') {{ e.preventDefault(); step(-1); }}
  if (e.key === 'ArrowRight') {{ e.preventDefault(); step(1); }}
  if (e.key === ' ') {{ e.preventDefault(); v.paused ? v.play() : v.pause(); }}
}});
update();
</script>
"""
components.html(html, height=height + plot_h + 70, scrolling=False)
st.caption("Click a trace to jump the video there. ← / → step between OD frames, "
           "space plays/pauses (click inside the player first). Detector pixels: " + "; ".join(notes))

# ----------------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------------
st.subheader("Export")
ts = pd.DataFrame({"Clock_time": clock, "Elapsed_s": t})
for k, y in traces.items():
    ts[("A" if k == "abs" else "I") + f"_{target:g}nm" + (f"_pm{half:g}" if half else "")] = y

mark_t = st.number_input("Mark a time on the exported figure (s, blank = none)",
                         min_value=0.0, value=None, step=0.1,
                         help="E.g. the moment of an event you saw in the video.")


def prism_fig():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.weight": "bold", "axes.labelweight": "bold"})
    fig, axes = plt.subplots(len(traces), 1, figsize=(7, 2.6 * len(traces) + 0.4), sharex=True)
    axes = np.atleast_1d(axes)
    for a, (k, y) in zip(axes, traces.items()):
        a.plot(t, y, color="k", lw=1.2)
        if show_pts:
            a.plot(t, y, "o", color="k", ms=2.5)
        if mark_t is not None:
            a.axvline(mark_t, color="red", ls=":", lw=2)
        a.set_ylabel(labels[k].replace(", ", "\n").replace(" (", "\n("))
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_linewidth(3.5)
        a.tick_params(direction="out", width=2.5, length=6, labelsize=11)
        for lab in a.get_xticklabels() + a.get_yticklabels():
            lab.set_fontweight("bold")
    axes[-1].set_xlabel("Experimental time (s)")
    fig.tight_layout()
    return fig


fig = prism_fig()
st.pyplot(fig)
svg, png = io.BytesIO(), io.BytesIO()
fig.savefig(svg, format="svg", bbox_inches="tight")
fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
plt.close(fig)
d1, d2, d3, d4 = st.columns(4)
d1.download_button("Traces (CSV)", ts.to_csv(index=False).encode(),
                   file_name=f"{stem}_{target:g}nm_traces.csv", mime="text/csv")
d2.download_button("Figure (SVG)", svg.getvalue(), file_name=f"{stem}_{target:g}nm_traces.svg",
                   mime="image/svg+xml")
d3.download_button("Figure (PNG, 300 dpi)", png.getvalue(),
                   file_name=f"{stem}_{target:g}nm_traces.png", mime="image/png")
d4.download_button("Converted video (MP4)", mp4,
                   file_name=os.path.splitext(vid_up.name)[0] + "_h264.mp4", mime="video/mp4")
