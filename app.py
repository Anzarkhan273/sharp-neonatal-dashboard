import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import streamlit as st

st.set_page_config(page_title="Hemodynamics Line + SD Dashboard", layout="wide")


# -----------------------------
# 1. Load and clean data
# -----------------------------
@st.cache_data
def load_data():
    """
    Try to load the CSV from:
    - Current directory (for Streamlit Cloud / GitHub)
    - Your local Windows path (for running on your laptop)
    """
    possible_paths = [
        "REDCAallPatientsDATA.csv",
        r"C:\Users\anzar\OneDrive\Desktop\hemo\REDCAallPatientsDATA.csv",
    ]

    df = None
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue

    if df is None:
        st.error(
            "Could not find 'REDCAallPatientsDATA.csv'.\n\n"
            "Make sure the file is either:\n"
            "- In the same folder as app.py in your GitHub repo, OR\n"
            "- At C:\\Users\\anzar\\OneDrive\\Desktop\\hemo\\REDCAallPatientsDATA.csv on your laptop."
        )
        st.stop()

    # Clean column names
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df


df = load_data()

# Numeric columns (for X and Y)
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# Grouping columns (for p-value) — small number of categories
group_cols = []
for c in df.columns:
    nunique = df[c].nunique(dropna=False)
    if 1 < nunique <= 15:
        group_cols.append(c)


# -----------------------------
# 2. Layout
# -----------------------------
st.title("Neonatal Hemodynamics – Line Plot with SD and p-value")

st.write(
    """
This app shows a **single line** (mean Y vs X) with an optional **±1 SD band**.

- Choose **X** (numeric) and **Y** (numeric) on the sidebar.
- Optionally turn on the **standard deviation band**.
- Optionally pick a **group** (with 2 levels) to compute a **p-value** for Y.

By default, only the **first 200 valid records** are used in this view, similar to your earlier setup.
"""
)


# -----------------------------
# 3. Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

if not numeric_cols:
    st.error("No numeric columns found in the dataset.")
    st.stop()

# Sensible defaults
default_x = "Day of Life" if "Day of Life" in numeric_cols else numeric_cols[0]
default_y = "SVC flow (mL/ kg/ min)" if "SVC flow (mL/ kg/ min)" in numeric_cols else numeric_cols[0]

x_col = st.sidebar.selectbox(
    "X-axis (numeric):",
    options=numeric_cols,
    index=numeric_cols.index(default_x) if default_x in numeric_cols else 0,
)

y_col = st.sidebar.selectbox(
    "Y-axis (numeric):",
    options=numeric_cols,
    index=numeric_cols.index(default_y) if default_y in numeric_cols else 0,
)

show_sd = st.sidebar.checkbox("Show standard deviation band", value=False)

group_for_pvalue = st.sidebar.selectbox(
    "Group (for p-value, optional):",
    options=["None"] + group_cols,
    index=0,
)

st.sidebar.markdown("---")
row_limit = st.sidebar.number_input(
    "Max records to use",
    min_value=50,
    max_value=len(df),
    value=min(200, len(df)),
    step=50,
)


# -----------------------------
# 4. Prepare data (first N rows)
# -----------------------------
cols = [x_col, y_col]
if group_for_pvalue != "None":
    cols.append(group_for_pvalue)

data = df[cols].dropna().head(int(row_limit))

if data.empty:
    st.warning("No data available for this combination of X, Y, and group.")
    st.stop()


# -----------------------------
# 5. Compute stats (median, mode, p-value)
# -----------------------------
y_series = data[y_col]
median_val = y_series.median()
modes = y_series.mode()
mode_val = modes.iloc[0] if len(modes) > 0 else np.nan

p_val = None
groups_used = None

if group_for_pvalue != "None":
    groups_used = data[group_for_pvalue].unique()
    if len(groups_used) == 2:
        g1 = data[data[group_for_pvalue] == groups_used[0]][y_col]
        g2 = data[data[group_for_pvalue] == groups_used[1]][y_col]
        if len(g1) > 1 and len(g2) > 1:
            _, p_val = ttest_ind(g1, g2, equal_var=False, nan_policy="omit")


# -----------------------------
# 6. Plot: mean line + optional SD band
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 5))

# Group by X in case there are repeated X values
grouped = data.groupby(x_col)[y_col]
x_vals = grouped.mean().index.values
y_mean = grouped.mean().values
y_sd = grouped.std().values  # SD per X (can be NaN if only 1 record at a given X)

# Main line (orange, solid)
ax.plot(
    x_vals,
    y_mean,
    linewidth=2.0,
    linestyle="-",
    color="#E69F00",  # orange
)

# Optional SD band
if show_sd:
    y_lower = y_mean - y_sd
    y_upper = y_mean + y_sd
    ax.fill_between(
        x_vals,
        y_lower,
        y_upper,
        alpha=0.2,
        color="#E69F00",
    )

# Label box at end of line
last_x = x_vals[-1]
last_y = y_mean[-1]
x_range = x_vals[-1] - x_vals[0] if len(x_vals) > 1 else 1
x_offset = 0.02 * x_range

ax.text(
    last_x + x_offset,
    last_y,
    y_col,
    va="center",
    fontsize=9,
    color="white",
    bbox=dict(
        boxstyle="round,pad=0.3",
        fc="#444444",
        ec="#111111",
        alpha=0.95,
    ),
)

# Extend x-limits so label isn't cut off
ax.set_xlim(x_vals[0], x_vals[-1] + 4 * x_offset)

# Labels & title
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.set_title(f"{y_col} vs {x_col} (first {len(data)} records)")

ax.grid(alpha=0.25)

plt.tight_layout()

# Show in Streamlit
st.pyplot(fig)


# -----------------------------
# 7. Show stats under the plot
# -----------------------------
st.markdown(
    f"""
**Records used in this view:** {len(data)}  
**Median of `{y_col}`:** {median_val:.2f}  
**Mode of `{y_col}`:** {mode_val:.2f}  
"""
)

if group_for_pvalue != "None":
    st.write(f"Groups in `{group_for_pvalue}` (in this view): {groups_used}")
    if p_val is not None:
        st.write(
            f"**p-value for `{y_col}` between `{groups_used[0]}` and `{groups_used[1]}`:** {p_val:.3f}"
        )
    else:
        st.write(
            "p-value not computed (need exactly 2 groups with enough data in each)."
        )


# -----------------------------
# 8. Download buttons
# -----------------------------
st.markdown("### Downloads")

# Data CSV
csv_bytes = data.to_csv(index=False).encode("utf-8")
csv_filename = f"{y_col.replace(' ', '_')}_vs_{x_col.replace(' ', '_')}.csv"

st.download_button(
    label="Download data (CSV)",
    data=csv_bytes,
    file_name=csv_filename,
    mime="text/csv",
)

# Plot JPG
img_buf = io.BytesIO()
fig.savefig(img_buf, format="jpg", dpi=300, bbox_inches="tight")
img_buf.seek(0)

img_filename = f"{y_col.replace(' ', '_')}_vs_{x_col.replace(' ', '_')}.jpg"

st.download_button(
    label="Download plot (JPG)",
    data=img_buf,
    file_name=img_filename,
    mime="image/jpeg",
)
