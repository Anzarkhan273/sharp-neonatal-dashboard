import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.colors as mcolors
import itertools
import io

st.set_page_config(page_title="Neonatal Dashboard", layout="wide")

MIN_N_PER_POINT = 10             # Shashank: omit points/categories with <10 samples
TITLE_PAD = 18                   # whitespace between title and top tick labels
DISCRETE_NUNIQUE_THRESHOLD = 20  # if X has <= this many unique values -> bar chart


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


def _mix_colors(c1, c2):
    """Average two colors -> mixed color for overlap."""
    r1 = np.array(mcolors.to_rgb(c1), dtype=float)
    r2 = np.array(mcolors.to_rgb(c2), dtype=float)
    return tuple((r1 + r2) / 2.0)


def _is_intlike(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if s.size == 0:
        return False
    return bool(np.all(np.isclose(s, np.round(s))))


def _is_discrete_x(df_plot: pd.DataFrame, xcol: str) -> bool:
    """
    Decide whether X should be treated as discrete (bar chart) vs continuous (line chart).
    Rule: if few unique values (<= threshold) OR integer-like binned data with modest range.
    """
    x = pd.to_numeric(df_plot[xcol], errors="coerce").dropna()
    nunique = int(x.nunique(dropna=True))

    if nunique <= DISCRETE_NUNIQUE_THRESHOLD:
        return True

    if _is_intlike(x):
        rng = float(x.max() - x.min()) if x.size else 0.0
        # if it's basically "binned" and not too wide, treat as discrete
        if nunique <= 50 and rng <= 80:
            return True

    return False


def _plot_series_line(
    ax,
    sub_df: pd.DataFrame,
    xcol: str,
    ycol: str,
    show_sd: bool,
    label: str | None,
    color: str | None = None,
    linestyle: str = "-",
    marker: str | None = None,
    min_n: int = MIN_N_PER_POINT,
):
    # Compute mean/std and also sample count N
    agg = (
        sub_df.groupby(xcol)[ycol]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .sort_values(xcol)
    )

    # Omit points with small sample size (DROP rows so the line stays connected)
    agg = agg[agg["n"] >= int(min_n)].copy()

    if agg.empty:
        return None

    xv = agg[xcol].to_numpy(dtype=float)
    ym = agg["mean"].to_numpy(dtype=float)
    ys = agg["std"].to_numpy(dtype=float)

    (ln,) = ax.plot(
        xv, ym,
        linewidth=2.5,
        label=label,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=4 if marker else 0,
        markevery=max(1, len(xv) // 12),
        zorder=3
    )

    line_color = ln.get_color() if color is None else color

    band_info = None
    if show_sd:
        lower = ym - ys
        upper = ym + ys
        finite = np.isfinite(xv) & np.isfinite(lower) & np.isfinite(upper)
        ax.fill_between(
            xv, lower, upper,
            where=finite,
            alpha=0.12,
            color=line_color,
            linewidth=0,
            zorder=1
        )
        band_info = {"x": xv, "lower": lower, "upper": upper, "color": line_color, "label": label}

    return band_info


def _plot_bars_with_sd(
    ax,
    df_plot: pd.DataFrame,
    xcol: str,
    ycol: str,
    gcol: str | None,
    show_sd: bool,
    bw_mode: bool,
    min_n: int = MIN_N_PER_POINT,
):
    """
    Bar chart for discrete X: mean with SD error bars.
    Omit categories where N < min_n (per group per category).
    """
    if gcol:
        stats = (
            df_plot.groupby([gcol, xcol])[ycol]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
        )
        stats = stats[stats["n"] >= int(min_n)].copy()
        if stats.empty:
            return

        x_vals = np.sort(stats[xcol].unique())
        groups = list(stats[gcol].dropna().unique())

        k = max(len(groups), 1)
        base = np.arange(len(x_vals))
        bar_w = 0.8 / k

        hatches = ["///", "\\\\\\", "xx", "..", "++", "--", "oo", "**"]

        for j, gv in enumerate(groups):
            sub = stats[stats[gcol] == gv].set_index(xcol)

            means = np.array([sub.at[x, "mean"] if x in sub.index else np.nan for x in x_vals], dtype=float)
            sds = np.array([sub.at[x, "std"] if x in sub.index else np.nan for x in x_vals], dtype=float)

            x_pos = base + (j - (k - 1) / 2.0) * bar_w
            finite = np.isfinite(means)

            if not np.any(finite):
                continue

            yerr = sds[finite] if show_sd else None

            if bw_mode:
                ax.bar(
                    x_pos[finite], means[finite],
                    width=bar_w,
                    yerr=yerr,
                    capsize=3 if show_sd else 0,
                    color="white",
                    edgecolor="black",
                    hatch=hatches[j % len(hatches)],
                    linewidth=1.0,
                    label=str(gv),
                    zorder=3
                )
            else:
                ax.bar(
                    x_pos[finite], means[finite],
                    width=bar_w,
                    yerr=yerr,
                    capsize=3 if show_sd else 0,
                    label=str(gv),
                    zorder=3
                )

        ax.set_xticks(base)
        if _is_intlike(pd.Series(x_vals)):
            ax.set_xticklabels([str(int(v)) for v in x_vals])
        else:
            ax.set_xticklabels([str(v) for v in x_vals])

    else:
        stats = (
            df_plot.groupby(xcol)[ycol]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
            .sort_values(xcol)
        )
        stats = stats[stats["n"] >= int(min_n)].copy()
        if stats.empty:
            return

        x_vals = stats[xcol].to_numpy(dtype=float)
        means = stats["mean"].to_numpy(dtype=float)
        sds = stats["std"].to_numpy(dtype=float)

        base = np.arange(len(x_vals))
        yerr = sds if show_sd else None

        if bw_mode:
            ax.bar(
                base, means,
                yerr=yerr,
                capsize=3 if show_sd else 0,
                color="white",
                edgecolor="black",
                linewidth=1.0,
                zorder=3
            )
        else:
            ax.bar(
                base, means,
                yerr=yerr,
                capsize=3 if show_sd else 0,
                zorder=3
            )

        ax.set_xticks(base)
        if _is_intlike(pd.Series(x_vals)):
            ax.set_xticklabels([str(int(v)) for v in x_vals])
        else:
            ax.set_xticklabels([str(v) for v in x_vals])

    # rotate if many categories
    if len(ax.get_xticklabels()) > 12:
        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_ha("right")


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


def _guess_y(cols, x_default):
    # Prefer SVC flow, but never return the same as X
    for c in cols:
        if ("SVC" in str(c)) and ("flow" in str(c)) and (c != x_default):
            return c
    # otherwise pick the first different column
    for c in cols:
        if c != x_default:
            return c
    return cols[0]


x_default = _guess_x(numeric_cols)
y_default = _guess_y(numeric_cols, x_default)

xcol = st.sidebar.selectbox("X-axis", numeric_cols, index=numeric_cols.index(x_default))
ycol = st.sidebar.selectbox("Y-axis", numeric_cols, index=numeric_cols.index(y_default))
bw_mode = st.sidebar.checkbox("B/W print mode (use line styles + markers)", value=False)

# --- Guardrail: stop cleanly if X and Y are the same (prevents the big traceback) ---
if xcol == ycol:
    st.warning("X and Y can’t be the same. Choose two different variables.")
    st.stop()
# -------------------------------------------------------------------------------

group_choice = st.sidebar.selectbox("Group (optional)", ["(None)"] + group_cols, index=0)
gcol = None if group_choice == "(None)" else group_choice

show_sd = st.sidebar.checkbox("Show SD", value=True)
show_legend = st.sidebar.checkbox("Show legend (outside)", value=True)

apply_axes = st.sidebar.checkbox("Apply axis controls", value=False)

# Eligibility first
needed = [xcol, ycol] + ([gcol] if gcol else [])
df_eligible = df_raw[needed].dropna()

rawN = int(df_raw.shape[0])
eligN = int(df_eligible.shape[0])

if eligN == 0:
    st.error("No rows have non-missing values for the selected fields.")
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

df_plot = df_eligible.sort_values(xcol, kind="mergesort").head(int(max_n)).copy()
shownN = int(df_plot.shape[0])

with st.sidebar.expander("Counts", expanded=False):
    st.write(f"Raw N: {rawN}")
    st.write(f"Eligible N (non-missing for selected columns): {eligN}")
    st.write(f"Shown N: {shownN}")

# Axis defaults
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

default_title = f"{ycol} vs {xcol}  (shown N={shownN} / eligible N={eligN} / raw N={rawN})"
plot_title = st.sidebar.text_input("Plot title", value=default_title)

# Decide plot type automatically:
# - discrete X -> bar chart with SD error bars
# - continuous X -> line chart with SD band
use_bar = _is_discrete_x(df_plot, xcol)

# ===== FIXED LAYOUT: plot + legend in separate fixed panels =====
fig = plt.figure(figsize=(11, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[4.8, 1.6], wspace=0.02)

ax = fig.add_subplot(gs[0, 0])       # main plot
ax_leg = fig.add_subplot(gs[0, 1])   # legend panel (fixed width)
ax_leg.axis("off")                   # hide legend panel axes

ax.grid(True, alpha=0.20)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

band_infos = []

if use_bar:
    # BAR CHART for discrete X (GA weeks, scores, categories)
    _plot_bars_with_sd(
        ax=ax,
        df_plot=df_plot,
        xcol=xcol,
        ycol=ycol,
        gcol=gcol,
        show_sd=show_sd,
        bw_mode=bw_mode,
        min_n=MIN_N_PER_POINT
    )
else:
    # LINE CHART for continuous X (time-like / continuous measurements)
    if gcol:
        linestyles = ["-", "--", ":", "-."]
        markers = ["o", "s", "^", "D", "X", "P", "v"]

        groups = list(df_plot.groupby(gcol))
        for i, (gval, gdf) in enumerate(groups):
            ls = linestyles[i % len(linestyles)]
            mk = markers[i % len(markers)]

            if bw_mode:
                info = _plot_series_line(
                    ax, gdf, xcol, ycol, show_sd, label=str(gval),
                    color="black", linestyle=ls, marker=mk,
                    min_n=MIN_N_PER_POINT
                )
            else:
                info = _plot_series_line(
                    ax, gdf, xcol, ycol, show_sd, label=str(gval),
                    color=None, linestyle=ls, marker=None,
                    min_n=MIN_N_PER_POINT
                )

            if info is not None:
                band_infos.append(info)

        # overlap shading (mixed color) only for line SD bands
        if show_sd and len(band_infos) >= 2:
            for a, b in itertools.combinations(band_infos, 2):
                dfa = pd.DataFrame({"x": a["x"], "la": a["lower"], "ua": a["upper"]})
                dfb = pd.DataFrame({"x": b["x"], "lb": b["lower"], "ub": b["upper"]})
                m = dfa.merge(dfb, on="x", how="inner").sort_values("x")
                if m.empty:
                    continue

                x = m["x"].to_numpy(dtype=float)
                lo = np.maximum(m["la"].to_numpy(dtype=float), m["lb"].to_numpy(dtype=float))
                hi = np.minimum(m["ua"].to_numpy(dtype=float), m["ub"].to_numpy(dtype=float))
                mask = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)

                if np.any(mask):
                    mix_c = "0.35" if bw_mode else _mix_colors(a["color"], b["color"])
                    ax.fill_between(x, lo, hi, where=mask, color=mix_c, alpha=0.22, linewidth=0, zorder=2)

    else:
        _plot_series_line(
            ax, df_plot, xcol, ycol, show_sd, label=None,
            color="black" if bw_mode else None, linestyle="-", marker=None,
            min_n=MIN_N_PER_POINT
        )

# Title padding fix
ax.set_title(plot_title, pad=TITLE_PAD)
ax.set_xlabel(xcol)
ax.set_ylabel(ycol)

if apply_axes:
    # For bar charts, X is categorical positions (0..k-1), so X axis numeric controls don't apply.
    # We still allow Y axis controls for both.
    errors = []
    if not (ymax > ymin):
        errors.append("Y max must be > Y min")
    if not (ystep > 0):
        errors.append("Y step must be > 0")

    if (not use_bar):
        if not (xmax > xmin):
            errors.append("X max must be > X min")
        if not (xstep > 0):
            errors.append("X step must be > 0")

    if errors:
        st.warning("Axis controls invalid:\n- " + "\n- ".join(errors))
    else:
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(np.arange(ymin, ymax + ystep, ystep))

        if not use_bar:
            ax.set_xlim(xmin, xmax)
            ax.set_xticks(np.arange(xmin, xmax + xstep, xstep))

# Legend goes in fixed right panel (so plot never changes width)
if gcol and show_legend:
    handles, labels = ax.get_legend_handles_labels()
    ax_leg.legend(handles, labels, title=gcol, loc="upper left", frameon=False, fontsize=9)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
buf.seek(0)

st.sidebar.download_button(
    label="Download plot (PNG)",
    data=buf,
    file_name="neonatal_plot.png",
    mime="image/png",
)

st.pyplot(fig, use_container_width=True)
