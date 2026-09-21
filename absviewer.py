"""
Live absorbance viewer
----------------------
Streamlit app for the *_abs.csv files written by spectro.m's "Start recording"
button. Each row of that file is one live frame:
    Clock_time, Elapsed_s, IntTime, NumAvg, <A at wavelength 1>, <A at wavelength 2>, ...
Absorbance is the signed value -log10((S-D)/(R-D)); NaN where undefined.

The app shows
  1. the absorbance spectrum of one frame (scrub frames with the slider), and
  2. absorbance vs. time at wavelengths you choose (e.g. 600 -> OD600),
and lets you download the plotted data (CSV) and each figure (SVG).
"""
import io
import re
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

META_COLS = ["Clock_time", "Elapsed_s", "IntTime", "NumAvg"]

st.set_page_config(page_title="Live absorbance viewer", layout="wide")
st.title("Live absorbance viewer")
st.caption("Upload the `_abs.csv` file from a spectro.m recording.")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading file...")
def load_recording(raw: bytes):
    df = pd.read_csv(io.BytesIO(raw))
    if "Wavelength_nm" in df.columns:
        raise ValueError(
            "This looks like the _meta.csv file (dark + reference). "
            "Upload the _abs.csv file instead."
        )
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected column(s): {', '.join(missing)}.")
    wl_cols = [c for c in df.columns if c not in META_COLS]
    try:
        wl = np.array([float(c) for c in wl_cols])
    except ValueError:
        raise ValueError("Wavelength column headers are not numeric.")
    A = df[wl_cols].to_numpy(dtype=np.float32)
    info = df[META_COLS].copy()
    info["Clock_time"] = info["Clock_time"].astype(str)
    return info, wl, A


def parse_wavelengths(text: str):
    vals = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", text)]
    return list(dict.fromkeys(vals))  # unique, keep order


def trace_at(wl, A, target, half_width):
    """Absorbance vs. frame at one wavelength.
    half_width == 0 -> nearest single pixel (same as the spectro.m OD boxes).
    half_width  > 0 -> mean of all pixels within target +/- half_width nm."""
    if half_width <= 0:
        i = int(np.argmin(np.abs(wl - target)))
        return A[:, i], f"{wl[i]:.1f} nm", f"A_{wl[i]:.2f}nm"
    sel = np.abs(wl - target) <= half_width
    if not sel.any():
        i = int(np.argmin(np.abs(wl - target)))
        return A[:, i], f"{wl[i]:.1f} nm", f"A_{wl[i]:.2f}nm"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows
        y = np.nanmean(A[:, sel], axis=1)
    lab = f"{target:g} \u00b1 {half_width:g} nm"
    return y, lab, f"A_{target:g}nm_pm{half_width:g}"


def fig_to_svg(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    return buf.getvalue()


def stem_of(name: str) -> str:
    s = re.sub(r"\.csv$", "", name, flags=re.I)
    return re.sub(r"_abs$", "", s)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    up = st.file_uploader("Recording file (_abs.csv)", type=["csv"])

if up is None:
    st.info("Upload a `_abs.csv` recording to begin.")
    st.stop()

if up.name.lower().endswith("_counts.csv"):
    st.warning(
        "This is the _counts.csv file (raw detector counts), so the plots "
        "below show counts, not absorbance. Upload the _abs.csv file for absorbance."
    )

try:
    info, wl, A = load_recording(up.getvalue())
except Exception as e:  # show a readable message instead of a traceback
    st.error(f"Could not read this file: {e}")
    st.stop()

n = A.shape[0]
t = info["Elapsed_s"].to_numpy(dtype=float)
stem = stem_of(up.name)
wl_min, wl_max = float(wl.min()), float(wl.max())

with st.sidebar:
    st.header("Time trace")
    wl_text = st.text_input(
        "Wavelength(s) to track (nm)", "600",
        help="One or more, separated by commas, e.g. 600, 700.",
    )
    half = st.number_input(
        "Average over \u00b1 nm", min_value=0.0, max_value=50.0, value=0.0, step=0.5,
        help="0 = the single pixel nearest each wavelength (same as the OD boxes "
             "in spectro.m). A value > 0 averages all pixels in that window.",
    )
    st.header("Spectrum plot")
    xr = st.slider(
        "Wavelength range (nm)", min_value=float(np.floor(wl_min)),
        max_value=float(np.ceil(wl_max)),
        value=(max(400.0, float(np.floor(wl_min))), min(900.0, float(np.ceil(wl_max)))),
        step=1.0,
    )
    overlay = st.checkbox("Overlay first frame", value=True)
    auto_y = st.checkbox("Auto y-axis", value=True)
    if not auto_y:
        c1, c2 = st.columns(2)
        ymin = c1.number_input("y min", value=-0.05, step=0.05, format="%.3f")
        ymax = c2.number_input("y max", value=0.5, step=0.05, format="%.3f")

targets = parse_wavelengths(wl_text)
bad = [w for w in targets if not (wl_min <= w <= wl_max)]
targets = [w for w in targets if wl_min <= w <= wl_max]
if bad:
    st.warning(
        f"Ignored wavelength(s) outside the sensor range "
        f"({wl_min:.1f}-{wl_max:.1f} nm): {', '.join(f'{b:g}' for b in bad)}"
    )

# ----------------------------------------------------------------------------
# Summary + frame slider
# ----------------------------------------------------------------------------
dt = np.diff(t)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Frames", f"{n}")
c2.metric("Duration", f"{t[-1] - t[0]:.2f} s")
c3.metric("Median frame interval", f"{np.median(dt):.3f} s" if n > 1 else "-")
c4.metric("Started", info["Clock_time"].iloc[0])

if n > 1:
    k = st.slider("Frame", min_value=1, max_value=n, value=1, step=1) - 1
else:
    k = 0
st.markdown(
    f"**Frame {k + 1} / {n}** &nbsp;·&nbsp; t = **{t[k]:.3f} s** "
    f"&nbsp;·&nbsp; clock {info['Clock_time'].iloc[k]}"
)

# ----------------------------------------------------------------------------
# Spectrum plot
# ----------------------------------------------------------------------------
xm = (wl >= xr[0]) & (wl <= xr[1])
fig1, ax1 = plt.subplots(figsize=(9, 4))
if overlay and k != 0:
    ax1.plot(wl[xm], A[0, xm], color="0.6", lw=1, label=f"Frame 1 (t = {t[0]:.2f} s)")
ax1.plot(wl[xm], A[k, xm], color="k", lw=1, label=f"Frame {k + 1} (t = {t[k]:.2f} s)")
for w in targets:
    ax1.axvline(w, color="tab:red", ls=":", lw=0.8)
ax1.set_xlim(xr)
if not auto_y:
    ax1.set_ylim(ymin, ymax)
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Absorbance")
ax1.set_title(f"{stem} - spectrum at t = {t[k]:.2f} s")
ax1.legend(loc="best", frameon=False, fontsize=9)
ax1.grid(alpha=0.25)
st.pyplot(fig1)

# ----------------------------------------------------------------------------
# Time plot
# ----------------------------------------------------------------------------
traces = []
fig2, ax2 = plt.subplots(figsize=(9, 3.6))
for w in targets:
    y, lab, col = trace_at(wl, A, w, half)
    traces.append((col, y))
    ax2.plot(t, y, marker="o", ms=2.5, lw=1, label=lab)
ax2.axvline(t[k], color="0.4", ls="--", lw=1)
ax2.set_xlabel("Elapsed time (s)")
ax2.set_ylabel("Absorbance")
ax2.set_title(f"{stem} - absorbance vs. time")
if targets:
    ax2.legend(loc="best", frameon=False, fontsize=9)
ax2.grid(alpha=0.25)
st.pyplot(fig2)

# ----------------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------------
st.subheader("Downloads")
ts = info[["Clock_time", "Elapsed_s"]].copy()
for col, y in traces:
    ts[col] = y
spec = pd.DataFrame({"Wavelength_nm": wl, f"A_frame{k + 1}_t{t[k]:.3f}s": A[k]})
if overlay and k != 0:
    spec[f"A_frame1_t{t[0]:.3f}s"] = A[0]

d1, d2, d3, d4 = st.columns(4)
d1.download_button(
    "Time trace (CSV)", ts.to_csv(index=False).encode(),
    file_name=f"{stem}_timetrace.csv", mime="text/csv", disabled=not traces,
)
d2.download_button(
    "Time plot (SVG)", fig_to_svg(fig2),
    file_name=f"{stem}_timetrace.svg", mime="image/svg+xml",
)
d3.download_button(
    f"Frame {k + 1} spectrum (CSV)", spec.to_csv(index=False).encode(),
    file_name=f"{stem}_spectrum_frame{k + 1}.csv", mime="text/csv",
)
d4.download_button(
    "Spectrum plot (SVG)", fig_to_svg(fig1),
    file_name=f"{stem}_spectrum_frame{k + 1}.svg", mime="image/svg+xml",
)
st.caption(
    "Spectrum CSV is the full sensor range; the plot shows only the selected "
    "wavelength range. Blank cells = NaN (absorbance undefined for that pixel)."
)
plt.close(fig1)
plt.close(fig2)
