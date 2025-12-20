import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Neonatal Dashboard", layout="wide")


def _nice_step(vmin: float, vmax: float) -> float:
    """Return a human-friendly tick step (~10 ticks across the axis)."""
    span = max(vmax - vmin, 1e-9)
    raw = span / 10.0
    k = 10 ** np.floor(np.log10(raw))
    candidates = np.array([1, 2, 5, 10], dtype=float) * k
    return float(candidates[np.argmin(np.abs(candidates - raw))])


def _compute_axis_defaults(df_plot: pd.DataFrame, xcol: str, ycol: str):
    xv = df_plot[xcol].to_numpy(dtype=float)
    yv = df_plot[ycol].to_numpy(dtype=float)
    xmin, xmax = float(np.nanmin(xv)), float(np.nanmax(xv))
    ymin, ymax = float(np.nanmin(yv)), float(np.nanmax(yv))
    return xmin, xmax, _nice_step(xmin, xmax), ymin, ymax, _nice_step(ymin, ymax)


def _load_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def _plot_series(
    ax,
    sub_df: pd.DataFrame,
    xcol: str,
    ycol: str,
    show_sd: bool,
    label: str | None,
    color: str | None = None,
    linestyle: str = "-",
    marker: str | None = None,
):
    agg = (
        sub_df.groupby(xcol)[ycol]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(xcol)
    )

    xv = agg[xcol].to_numpy(dtype=float)
    ym = agg["mean"].to_numpy(dtype=float)
    ys = agg["std"].to_numpy(dtype=float)

    ax.plot(
        xv, ym,
        linewidth=2.5,
        label=label,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=4 if marker else 0,
        markevery=max(1, len(xv)//12)  # keeps markers readable
    )

    if show_sd:
        ax.fill_between(xv, ym - ys, ym + ys, alpha=0.12, color=color)



st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Upload your CSV (e.g., REDCAallPatientsDATA.csv) to begin.")
    st.stop()

df_raw = _load_csv(uploaded)

numeric_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
if len(numeric_cols) < 2:
    st.error("Need at least 2 numeric columns to plot.")
    st.stop()

group_cols = [c for c in df_raw.columns if 1 < df_raw[c].nunique(dropna=False) <= 15]


def _guess_x(cols):
    for c in cols:
        if "Gestation" in str(c):
            return c
    return cols[0]


def _guess_y(cols):
    for c in cols:
        if ("SVC" in str(c)) and ("flow" in str(c)):
            return c
    return cols[1] if len(cols) > 1 else cols[0]


xcol = st.sidebar.selectbox("X-axis", numeric_cols, index=numeric_cols.index(_guess_x(numeric_cols)))
ycol = st.sidebar.selectbox("Y-axis", numeric_cols, index=numeric_cols.index(_guess_y(numeric_cols)))

group_choice = st.sidebar.selectbox("Group (optional)", ["(None)"] + group_cols, index=0)
gcol = None if group_choice == "(None)" else group_choice

show_sd = st.sidebar.checkbox("Show SD band", value=True)
show_legend = st.sidebar.checkbox("Show legend (outside)", value=True)

apply_axes = st.sidebar.checkbox("Apply axis controls", value=False)

# Eligibility first (explains your 'stuck at 462' issue)
needed = [xcol, ycol] + ([gcol] if gcol else [])
df_eligible = df_raw[needed].dropna()

rawN = int(df_raw.shape[0])
eligN = int(df_eligible.shape[0])

if eligN == 0:
    st.error("No rows have non-missing values for your selected X/Y (and Group, if chosen).")
    st.stop()

# Max N slider reflects what can actually be shown
max_cap = eligN
step = 50 if max_cap >= 50 else 1
default_val = min(500, max_cap)

max_n = st.sidebar.slider(
    "Max records (N)",
    min_value=1,
    max_value=max_cap,
    value=default_val,
    step=step,
)

# Max N is based on X ordering (Record ID behavior)
df_plot = df_eligible.sort_values(xcol, kind="mergesort").head(int(max_n)).copy()
shownN = int(df_plot.shape[0])

with st.sidebar.expander("Counts", expanded=False):
    st.write(f"Raw N: {rawN}")
    st.write(f"Eligible N (non-missing for selected columns): {eligN}")
    st.write(f"Shown N: {shownN}")

# Axis defaults that auto-fill, but keep user edits
xmin_d, xmax_d, xstep_d, ymin_d, ymax_d, ystep_d = _compute_axis_defaults(df_plot, xcol, ycol)

axis_key = f"{xcol}|{ycol}|{gcol}|{shownN}|{eligN}"
if st.session_state.get("axis_key") != axis_key:
    st.session_state["axis_key"] = axis_key
    st.session_state["xmin"] = xmin_d
    st.session_state["xmax"] = xmax_d
    st.session_state["xstep"] = xstep_d
    st.session_state["ymin"] = ymin_d
    st.session_state["ymax"] = ymax_d
    st.session_state["ystep"] = ystep_d

with st.sidebar.expander("Axis controls (auto-filled defaults)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        xmin = st.number_input("X min", value=float(st.session_state["xmin"]))
        xmax = st.number_input("X max", value=float(st.session_state["xmax"]))
        xstep = st.number_input("X step", min_value=0.0, value=float(st.session_state["xstep"]))
    with c2:
        ymin = st.number_input("Y min", value=float(st.session_state["ymin"]))
        ymax = st.number_input("Y max", value=float(st.session_state["ymax"]))
        ystep = st.number_input("Y step", min_value=0.0, value=float(st.session_state["ystep"]))

    st.session_state["xmin"] = xmin
    st.session_state["xmax"] = xmax
    st.session_state["xstep"] = xstep
    st.session_state["ymin"] = ymin
    st.session_state["ymax"] = ymax
    st.session_state["ystep"] = ystep

    if st.button("Reset axis defaults"):
        st.session_state["xmin"] = xmin_d
        st.session_state["xmax"] = xmax_d
        st.session_state["xstep"] = xstep_d
        st.session_state["ymin"] = ymin_d
        st.session_state["ymax"] = ymax_d
        st.session_state["ystep"] = ystep_d
        st.rerun()

st.title("Neonatal Hemodynamics Dashboard")

fig, ax = plt.subplots(figsize=(11, 5))
ax.grid(True, alpha=0.20)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

if gcol:
    linestyles = ["-", "--", ":", "-."]              # solid, dashed, dotted, dash-dot
    markers = ["o", "s", "^", "D", "X", "P", "v"]     # circle, square, triangle, etc.

    groups = list(df_plot.groupby(gcol))
    for i, (gval, gdf) in enumerate(groups):
        ls = linestyles[i % len(linestyles)]
        mk = markers[i % len(markers)]

        # In B/W mode: force black and use linestyle+marker to differentiate
        if bw_mode:
            _plot_series(ax, gdf, xcol, ycol, show_sd, label=str(gval),
                         color="black", linestyle=ls, marker=mk)
        else:
            # In color mode: keep default matplotlib colors, still add linestyle for clarity
            _plot_series(ax, gdf, xcol, ycol, show_sd, label=str(gval),
                         color=None, linestyle=ls, marker=None)
else:
    _plot_series(ax, df_plot, xcol, ycol, show_sd, label=None,
                 color="black" if bw_mode else None, linestyle="-", marker=None)


ax.set_title(f"{ycol} vs {xcol}  (shown N={shownN} / eligible N={eligN} / raw N={rawN})")
ax.set_xlabel(xcol)
ax.set_ylabel(ycol)

if apply_axes:
    errors = []
    if not (xmax > xmin):
        errors.append("X max must be > X min")
    if not (ymax > ymin):
        errors.append("Y max must be > Y min")
    if not (xstep > 0):
        errors.append("X step must be > 0")
    if not (ystep > 0):
        errors.append("Y step must be > 0")

    if errors:
        st.warning("Axis controls invalid:\n- " + "\n- ".join(errors))
    else:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(np.arange(xmin, xmax + xstep, xstep))
        ax.set_yticks(np.arange(ymin, ymax + ystep, ystep))

# Always reserve the same right-side space so plot size stays consistent
fig.subplots_adjust(left=0.08, right=0.82, top=0.88, bottom=0.18)

# Legend outside (optional) — does not change plot size now
if gcol and show_legend:
    ax.legend(title=gcol, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)


st.pyplot(fig, use_container_width=False)




