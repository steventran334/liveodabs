"""
Live absorbance viewer
----------------------
Streamlit app for the files written by spectro.m's "Start recording" button:
    <name>_abs.csv     signed absorbance per live frame  -log10((S-D)/(R-D)), NaN where undefined
    <name>_counts.csv  raw sample intensity (counts) per live frame
    <name>_meta.csv    dark + reference spectra captured before the recording
The _abs and _counts files share one layout, one row per frame:
    Clock_time, Elapsed_s, IntTime, NumAvg, <value at wavelength 1>, <value at wavelength 2>, ...

For each of absorbance and intensity the app shows
  1. the spectrum of one frame (scrub frames with the slider), and
  2. the value vs. time at wavelengths you choose (e.g. 600 -> OD600),
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
st.caption(
    "Upload the files from one spectro.m recording: `_abs.csv` for absorbance, "
    "`_counts.csv` for intensity, and optionally `_meta.csv` to overlay the dark "
    "and reference spectra on the intensity plot."
)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading file...")
def read_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


def parse_frames(df: pd.DataFrame):
    """Split a frame file (_abs or _counts) into info, wavelengths, values."""
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing expected column(s): {', '.join(missing)}")
    wl_cols = [c for c in df.columns if c not in META_COLS]
    try:
        wl = np.array([float(c) for c in wl_cols])
    except ValueError:
        raise ValueError("wavelength column headers are not numeric")
    vals = df[wl_cols].to_numpy(dtype=np.float32)
    info = df[META_COLS].copy()
    info["Clock_time"] = info["Clock_time"].astype(str)
    return info, wl, vals


def classify(name: str, df: pd.DataFrame) -> str:
    """Return 'abs', 'counts' or 'meta' for an uploaded file."""
    if "Wavelength_nm" in df.columns:
        return "meta"
    low = name.lower()
    if low.endswith("_abs.csv"):
        return "abs"
    if low.endswith("_counts.csv"):
        return "counts"
    # Renamed file: absorbance values are small, raw counts are large.
    vals = df.drop(columns=[c for c in META_COLS if c in df.columns]).to_numpy(float)
    return "counts" if np.nanmedian(np.abs(vals)) > 50 else "abs"


def parse_wavelengths(text: str):
    vals = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", text)]
    return list(dict.fromkeys(vals))  # unique, keep order


def trace_at(wl, V, target, half_width, prefix):
    """Value vs. frame at one wavelength.
    half_width == 0 -> nearest single pixel (same as the spectro.m OD boxes).
    half_width  > 0 -> mean of all pixels within target +/- half_width nm.
    Labels use the wavelength the user typed (e.g. "600 nm"); the actual
    pixel wavelength(s) used are returned separately as `note`."""
    i = int(np.argmin(np.abs(wl - target)))
    sel = np.abs(wl - target) <= half_width
    if half_width <= 0 or not sel.any():
        note = f"{target:g} nm -> pixel at {wl[i]:.3f} nm"
        return V[:, i], f"{target:g} nm", f"{prefix}_{target:g}nm", note
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows
        y = np.nanmean(V[:, sel], axis=1)
    note = (f"{target:g} \u00b1 {half_width:g} nm -> mean of {int(sel.sum())} pixels "
            f"({wl[sel].min():.3f}-{wl[sel].max():.3f} nm)")
    return (y, f"{target:g} \u00b1 {half_width:g} nm",
            f"{prefix}_{target:g}nm_pm{half_width:g}", note)


def fig_to_svg(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    return buf.getvalue()


def stem_of(name: str) -> str:
    s = re.sub(r"\.csv$", "", name, flags=re.I)
    return re.sub(r"_(abs|counts|meta)$", "", s, flags=re.I)


# ----------------------------------------------------------------------------
# Upload
# ----------------------------------------------------------------------------
with st.sidebar:
    ups = st.file_uploader(
        "Recording file(s)", type=["csv"], accept_multiple_files=True,
        help="Any of _abs.csv, _counts.csv, _meta.csv from the same recording.",
    )

if not ups:
    st.info("Upload a `_abs.csv` and/or `_counts.csv` recording to begin.")
    st.stop()

files = {}
for u in ups:
    try:
        df = read_csv(u.getvalue())
        kind = classify(u.name, df)
    except Exception as e:
        st.error(f"Could not read {u.name}: {e}")
        st.stop()
    if kind in files:
        st.error(f"Two {kind} files uploaded ({files[kind][0]} and {u.name}). "
                 "Upload one recording at a time.")
        st.stop()
    files[kind] = (u.name, df)

if "abs" not in files and "counts" not in files:
    st.error("Only the _meta.csv file was uploaded. Add the _abs.csv and/or "
             "_counts.csv file from the same recording.")
    st.stop()

data = {}
for kind in ("abs", "counts"):
    if kind in files:
        try:
            data[kind] = parse_frames(files[kind][1])
        except Exception as e:
            st.error(f"Could not read {files[kind][0]}: {e}")
            st.stop()

# Both frame files must come from the same recording.
if len(data) == 2:
    (ia, wa, Va), (ic, wc, Vc) = data["abs"], data["counts"]
    same = (Va.shape == Vc.shape and np.allclose(wa, wc)
            and np.allclose(ia["Elapsed_s"], ic["Elapsed_s"]))
    if not same:
        st.error("The _abs and _counts files don't match (different frames or "
                 "wavelengths). Upload files from the same recording.")
        st.stop()

meta = None
if "meta" in files:
    m = files["meta"][1]
    need = {"Wavelength_nm", "Dark_counts", "Reference_counts"}
    if need.issubset(m.columns):
        meta = m
    else:
        st.warning("The _meta.csv file is missing expected columns; ignoring it.")

first_kind = "abs" if "abs" in data else "counts"
info, wl, _ = data[first_kind]
n = len(info)
t = info["Elapsed_s"].to_numpy(dtype=float)
stem = stem_of(files[first_kind][0])
wl_min, wl_max = float(wl.min()), float(wl.max())

if meta is not None and (len(meta) != len(wl) or
                         not np.allclose(meta["Wavelength_nm"].to_numpy(float), wl)):
    st.warning("The _meta.csv wavelengths don't match the recording; ignoring it.")
    meta = None

# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Time traces")
    wl_text = st.text_input(
        "Wavelength(s) to track (nm)", "600",
        help="One or more, separated by commas, e.g. 600, 700.",
    )
    half = st.number_input(
        "Average over \u00b1 nm", min_value=0.0, max_value=50.0, value=0.0, step=0.5,
        help="0 = the single pixel nearest each wavelength (same as the OD boxes "
             "in spectro.m). A value > 0 averages all pixels in that window.",
    )
    st.header("Spectrum plots")
    xr = st.slider(
        "Wavelength range (nm)", min_value=float(np.floor(wl_min)),
        max_value=float(np.ceil(wl_max)),
        value=(max(400.0, float(np.floor(wl_min))), min(900.0, float(np.ceil(wl_max)))),
        step=1.0,
    )
    overlay = st.checkbox("Overlay first frame", value=True)
    ylims = {}
    if "abs" in data:
        if not st.checkbox("Auto y-axis (absorbance)", value=True):
            c1, c2 = st.columns(2)
            ylims["abs"] = (c1.number_input("A min", value=-0.05, step=0.05, format="%.3f"),
                            c2.number_input("A max", value=0.5, step=0.05, format="%.3f"))
    if "counts" in data:
        show_dr = False
        if meta is not None:
            show_dr = st.checkbox("Show dark + reference on intensity plot", value=True)
        if not st.checkbox("Auto y-axis (intensity)", value=True):
            c1, c2 = st.columns(2)
            ylims["counts"] = (c1.number_input("Counts min", value=0.0, step=500.0),
                               c2.number_input("Counts max", value=16000.0, step=500.0))

targets = parse_wavelengths(wl_text)
bad = [w for w in targets if not (wl_min <= w <= wl_max)]
targets = [w for w in targets if wl_min <= w <= wl_max]
if bad:
    st.warning(
        f"Ignored wavelength(s) outside the sensor range "
        f"({wl_min:.1f}-{wl_max:.1f} nm): {', '.join(f'{b:g}' for b in bad)}"
    )

# ----------------------------------------------------------------------------
# Summary + frame slider (shared by all plots)
# ----------------------------------------------------------------------------
dt = np.diff(t)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Frames", f"{n}")
c2.metric("Duration", f"{t[-1] - t[0]:.2f} s")
c3.metric("Median frame interval", f"{np.median(dt):.3f} s" if n > 1 else "-")
c4.metric("Started", info["Clock_time"].iloc[0])

k = st.slider("Frame", min_value=1, max_value=n, value=1, step=1) - 1 if n > 1 else 0
st.markdown(
    f"**Frame {k + 1} / {n}** &nbsp;·&nbsp; t = **{t[k]:.3f} s** "
    f"&nbsp;·&nbsp; clock {info['Clock_time'].iloc[k]}"
)
xm = (wl >= xr[0]) & (wl <= xr[1])


def section(kind, V, ylabel, prefix, title_word, extra_lines=()):
    """Spectrum plot + time plot + downloads for one data type."""
    # --- spectrum at the selected frame
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    for lab, y, color in extra_lines:
        ax1.plot(wl[xm], y[xm], color=color, lw=1, alpha=0.8, label=lab)
    if overlay and k != 0:
        ax1.plot(wl[xm], V[0, xm], color="0.6", lw=1, label=f"Frame 1 (t = {t[0]:.2f} s)")
    ax1.plot(wl[xm], V[k, xm], color="k", lw=1, label=f"Frame {k + 1} (t = {t[k]:.2f} s)")
    for w in targets:
        ax1.axvline(w, color="tab:red", ls=":", lw=0.8)
    ax1.set_xlim(xr)
    if kind in ylims:
        ax1.set_ylim(*ylims[kind])
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel(ylabel)
    ax1.set_title(f"{stem} - {title_word} spectrum at t = {t[k]:.2f} s")
    ax1.legend(loc="best", frameon=False, fontsize=9)
    ax1.grid(alpha=0.25)
    st.pyplot(fig1)

    # --- value vs. time at the chosen wavelengths
    traces, notes = [], []
    fig2, ax2 = plt.subplots(figsize=(9, 3.6))
    for w in targets:
        y, lab, col, note = trace_at(wl, V, w, half, prefix)
        traces.append((col, y))
        notes.append(note)
        ax2.plot(t, y, marker="o", ms=2.5, lw=1, label=lab)
    ax2.axvline(t[k], color="0.4", ls="--", lw=1)
    ax2.set_xlabel("Elapsed time (s)")
    ax2.set_ylabel(ylabel)
    ax2.set_title(f"{stem} - {title_word} vs. time")
    if targets:
        ax2.legend(loc="best", frameon=False, fontsize=9)
    ax2.grid(alpha=0.25)
    st.pyplot(fig2)
    if notes:
        st.caption("Detector pixels used: " + "; ".join(notes))

    # --- downloads
    ts = info[["Clock_time", "Elapsed_s"]].copy()
    for col, y in traces:
        ts[col] = y
    spec = pd.DataFrame({"Wavelength_nm": wl, f"{prefix}_frame{k + 1}_t{t[k]:.3f}s": V[k]})
    if overlay and k != 0:
        spec[f"{prefix}_frame1_t{t[0]:.3f}s"] = V[0]
    for lab, y, _ in extra_lines:
        spec[lab.replace(" ", "_") + "_counts"] = y

    d1, d2, d3, d4 = st.columns(4)
    d1.download_button(
        "Time trace (CSV)", ts.to_csv(index=False).encode(),
        file_name=f"{stem}_{kind}_timetrace.csv", mime="text/csv",
        disabled=not traces, key=f"{kind}_ts_csv",
    )
    d2.download_button(
        "Time plot (SVG)", fig_to_svg(fig2),
        file_name=f"{stem}_{kind}_timetrace.svg", mime="image/svg+xml",
        key=f"{kind}_ts_svg",
    )
    d3.download_button(
        f"Frame {k + 1} spectrum (CSV)", spec.to_csv(index=False).encode(),
        file_name=f"{stem}_{kind}_spectrum_frame{k + 1}.csv", mime="text/csv",
        key=f"{kind}_sp_csv",
    )
    d4.download_button(
        "Spectrum plot (SVG)", fig_to_svg(fig1),
        file_name=f"{stem}_{kind}_spectrum_frame{k + 1}.svg", mime="image/svg+xml",
        key=f"{kind}_sp_svg",
    )
    plt.close(fig1)
    plt.close(fig2)


# ----------------------------------------------------------------------------
# Absorbance section
# ----------------------------------------------------------------------------
if "abs" in data:
    st.subheader("Absorbance")
    section("abs", data["abs"][2], "Absorbance", "A", "absorbance")
else:
    st.info("Upload the `_abs.csv` file to see absorbance.")

# ----------------------------------------------------------------------------
# Intensity section
# ----------------------------------------------------------------------------
if "counts" in data:
    st.subheader("Intensity")
    extra = []
    if meta is not None and show_dr:
        extra = [("Dark", meta["Dark_counts"].to_numpy(float), "tab:blue"),
                 ("Reference", meta["Reference_counts"].to_numpy(float), "tab:red")]
    section("counts", data["counts"][2], "Intensity (counts)", "I", "intensity",
            extra_lines=extra)
else:
    st.info("Upload the `_counts.csv` file to see intensity.")

st.caption(
    "Spectrum CSVs cover the full sensor range; the plots show only the selected "
    "wavelength range. Blank cells = NaN (absorbance undefined for that pixel)."
)
