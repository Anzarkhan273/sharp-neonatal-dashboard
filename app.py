import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import streamlit as st

st.set_page_config(page_title="Hemodynamics Dashboard", layout="wide")


# -----------------------------
# 1. Load and clean data
# -----------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

uploaded_file = st.file_uploader("Upload REDCA CSV", type=["csv"])
df = load_data(uploaded_file)

if df is None:
    st.stop()

numeric_cols = df.select_dtypes(include="number").columns.tolist()

group_cols = []
for c in df.columns:
    nunique = df[c].nunique(dropna=False)
    if 1 < nunique <= 15:
        group_cols.append(c)


# -----------------------------
# 2. Streamlit page layout
# -----------------------------
st.title("Neonatal Hemodynamics Dashboard")

st.write(
    """
This app is adapted from my original Jupyter notebook.

Select a metric and a grouping variable. The plot shows the metric by group, and
the box on the right displays **median**, **mode**, and **p-value** (if there are exactly two groups).
"""
)


# -----------------------------
# 3. Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

if numeric_cols:
    if "SVC flow (mL/ kg/ min)" in numeric_cols:
        default_y_index = numeric_cols.index("SVC flow (mL/ kg/ min)")
    else:
        default_y_index = 0

    y_col = st.sidebar.selectbox(
        "Y (metric):",
        options=numeric_cols,
        index=default_y_index,
    )
else:
    st.error("No numeric columns found in the dataset.")
    st.stop()

if group_cols:
    if "Grade of IVH LEFT" in group_cols:
        default_g_index = group_cols.index("Grade of IVH LEFT")
    else:
        default_g_index = 0

    group_col = st.sidebar.selectbox(
        "Group by:",
        options=group_cols,
        index=default_g_index,
    )
else:
    st.error("No suitable grouping columns found in the dataset.")
    st.stop()

st.sidebar.markdown("---")
row_limit = st.sidebar.number_input(
    "Max rows to use",
    min_value=100,
    max_value=len(df),
    value=min(200, len(df)),
    step=100,
)


# -----------------------------
# 4. Prepare the data
# -----------------------------
data = df[[y_col, group_col]].dropna().head(int(row_limit))

if data.empty:
    st.warning("No data available for this combination.")
    st.stop()


# -----------------------------
# 5. Compute stats
# -----------------------------
series = data[y_col].dropna()

if series.empty:
    median_val = np.nan
    mode_val = np.nan
else:
    median_val = series.median()
    modes = series.mode()
    mode_val = modes.iloc[0] if len(modes) > 0 else np.nan

groups = data[group_col].dropna().unique()
p_val = None

if len(groups) == 2:
    g1 = data[data[group_col] == groups[0]][y_col].dropna()
    g2 = data[data[group_col] == groups[1]][y_col].dropna()
    if len(g1) > 1 and len(g2) > 1:
        _, p_val = ttest_ind(
            g1,
            g2,
            equal_var=False,
            nan_policy="omit",
        )


# -----------------------------
# 6. Plot
# -----------------------------
fig, ax = plt.subplots()

for name, grp in data.groupby(group_col):
    ax.plot(
        grp.index,
        grp[y_col],
        marker="o",
        linestyle="-",
        label=str(name),
    )

ax.set_xlabel("Index")
ax.set_ylabel(y_col)
ax.set_title(f"{y_col} by {group_col}")
ax.legend(title=group_col)

stats_lines = [
    f"Median: {median_val:.2f}",
    f"Mode: {mode_val:.2f}",
]
if p_val is not None:
    stats_lines.append(f"p-value ({groups[0]} vs {groups[1]}): {p_val:.3f}")

ax.text(
    1.02,
    0.98,
    "\n".join(stats_lines),
    transform=ax.transAxes,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round", alpha=0.3),
)

plt.tight_layout()
st.pyplot(fig)


# -----------------------------
# 7. Download buttons
# -----------------------------
st.markdown("### Downloads")

csv_bytes = data.to_csv(index=False).encode("utf-8")
csv_filename = f"{y_col.replace(' ', '_')}_by_{group_col.replace(' ', '_')}.csv"

st.download_button(
    label="Download data (CSV)",
    data=csv_bytes,
    file_name=csv_filename,
    mime="text/csv",
)

img_buf = io.BytesIO()
fig.savefig(img_buf, format="jpg", dpi=300, bbox_inches="tight")
img_buf.seek(0)

img_filename = f"{y_col.replace(' ', '_')}_by_{group_col.replace(' ', '_')}.jpg"

st.download_button(
    label="Download plot (JPG)",
    data=img_buf,
    file_name=img_filename,
    mime="image/jpeg",
)

