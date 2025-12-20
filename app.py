import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Neonatal Dashboard", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def nice_step(vmin: float, vmax: float) -> float:
    span = max(vmax - vmin, 1e-9)
    raw = span / 10.0
    k = 10 ** np.floor(np.log10(raw))
    candidates = np.array([1, 2, 5, 10], dtype=float) * k
    return float(candidates[np.argmin(np.abs(candidates - raw))])

def compute_axis_defaults(df_plot: pd.DataFrame, xcol: str, ycol: str):
    xv = df_plot[xcol].to_numpy(dtype=float)
    yv = df_plot[ycol].to_numpy(dtype=float)
    xmin, xmax = float(np.nanmin(xv)), float(np.nanmax(xv))
    ymin, ymax = float(np.nanmin(yv)), float(np.nanmax(yv))
    return xmin, xmax, nice_step(xmin, xmax), ymin, ymax, nice_step(ymin, ymax)

def load_csv(uploaded_file):
    if uploaded_file is None:
        return None
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def plot_series(ax, sub_df: pd.DataFrame, xcol: str, ycol: str, show_sd: bool, label: str | None):
    agg = (
        sub_df.groupby(xcol)[ycol]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(xcol)
    )
    xv = agg[xcol].to_numpy(dtype=float)
    ym = agg["mean"].to_numpy(dtype=float)
    ys = agg["std"].to_numpy(dtype=float)

    ax.plot(xv, ym, linewidth=2.5, label=label)
    if show_sd:
        ax.fill_between(xv, ym - ys, ym + ys, alpha=0.12)

# ----------------------------
# Sidebar: Upload + controls
# ----------------------------
st.sidebar.header("Controls")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.caption("Tip: Upload the same REDCAallPatientsDATA.csv you used locally.")
df_raw = load_csv(uploaded)

if df_raw is None:
    st.warning("Upload a CSV to begin.")
    st.stop()

# Numeric & groupable cols
numeric_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
if len(numeric_cols) < 2:
    st.error("Need at least 2 numeric columns to plot.")
    st.stop()

group_cols = [c for c in df_raw.columns if 1 < df_raw[c].nunique(dropna=False) <= 15]

def guess_x(cols):
    for c in cols:
        if "Gestation" in str(c):
            return c
    return cols[0]

def guess_y(cols):
    for c in cols:
        if ("SVC" in str(c)) and ("flow" in str(c)):
            return c
    return cols[1] if len(cols) > 1 else cols[0]

# Basic selectors
xcol = st.sidebar.selectbox("X-axis", numeric_cols, index=numeric_cols.index(guess_x(numeric_cols)))
ycol = st.sidebar.selectbox("Y-axis", numeric_cols, index=numeric_cols.index(guess_y(numeric_cols)))

group_choice = st.sidebar.selectbox("Group (optional)", ["(None)"] + group_cols, index=0)
gcol = None if group_choice == "(None)" else group_choice

show_sd = st.sidebar.checkbox("Show SD band", value=True)
show_legend = st.sidebar.checkbox("Show legend (outside)", value=True)

max_n = st.sidebar.slider("Max records (N)", min_value=50, max_value=int(len(df_raw)), value=min(500, int(len(df_raw))), step=50)

apply_axes = st.sidebar.checkbox("Apply axis controls", value=False)

# ----------------------------
# Build df_eligible & df_plot (Max N behaves by X sorting)
# ----------------------------
needed = [xcol, ycol] + ([gcol] if gcol else [])
df_eligible = df_raw[needed].dropna()

# Key fix: Max N is based on X ordering, not file order
df_plot = df_eligible.sort_values(xcol, kind="mergesort").head(int(max_n)).copy()

rawN = df_raw.shape[0]
eligN = df_eligible.shape[0]
shownN = df_plot.shape[0]

if shownN == 0:
    st.error("No rows to plot after dropping missing values for selected columns.")
    st.stop()

# ----------------------------
# Axis defaults + session state (so defaults are populated, but user edits persist)
# ----------------------------
xmin_d, xmax_d, xstep_d, ymin_d, ymax_d, ystep_d = compute_axis_defaults(df_plot, xcol, ycol)

axis_key = f"{xcol}|{ycol}|{gcol}|{max_n}|{shownN}"
if st.session_state.get("axis_key") != axis_key:
    # Update defaults when graph context changes
    st.session_state["axis_key"] = axis_key
    st.session_state["xmin"] = xmin_d
    st.session_state["xmax"] = xmax_d
    st.session_state["xstep"] = xstep_d
    st.session_state["ymin"] = ymin_d
    st.session_state["ymax"] = ymax_d
    st.session_state["ystep"] = ystep_d

with st.sidebar.expander("Axis controls (auto-filled defaults)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        xmin = st.number_input("X min", value=float(st.session_state["xmin"]))
        xmax = st.number_input("X max", value=float(st.session_state["xmax"]))
        xstep = st.number_input("X step", min_value=0.0, value=float(st.session_state["xstep"]))
    with col2:
        ymin = st.number_input("Y min", value=float(st.session_state["ymin"]))
        ymax = st.number_input("Y max", value=float(st.session_state["ymax"]))
        ystep = st.number_input("Y step", min_value=0.0, value=float(st.session_state["ystep"]))

    # Persist user edits
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

# ----------------------------
# Plot
# ----------------------------
st.title("Neonatal Hemodynamics Dashboard")

fig, ax = plt.subplots(figsize=(11, 5))
ax.grid(True, alpha=0.20)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

if gcol:
    for gval, gdf in df_plot.groupby(gcol):
        plot_series(ax, gdf, xcol, ycol, show_sd, label=str(gval))
else:
    plot_series(ax, df_plot, xcol, ycol, show_sd, label=None)

ax.set_title(f"{ycol} vs {xcol}  (shown N={shownN} / eligible N={eligN} / raw N={rawN})")
ax.set_xlabel(xcol)
ax.set_ylabel(ycol)

# Apply axis controls
if apply_axes:
    errors = []
    if not (xmax > xmin): errors.append("X max must be > X min")
    if not (ymax > ymin): errors.append("Y max must be > Y min")
    if not (xstep > 0): errors.append("X step must be > 0")
    if not (ystep > 0): errors.append("Y step must be > 0")

    if errors:
        st.warning("Axis controls invalid:\n- " + "\n- ".join(errors))
    else:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(np.arange(xmin, xmax + xstep, xstep))
        ax.set_yticks(np.arange(ymin, ymax + ystep, ystep))

# Legend outside (not on top)
if gcol and show_legend:
    ax.legend(title=gcol, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
else:
    fig.tight_layout()

st.pyplot(fig)
