import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown, Checkbox
from scipy.stats import ttest_ind

plt.style.use("default")


# -----------------------------
# 1. Load data and detect columns
# -----------------------------
path = r"C:\Users\anzar\OneDrive\Desktop\hemo\REDCAallPatientsDATA.csv"

df = pd.read_csv(path)

# Clean column names
df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
df = df.loc[:, ~df.columns.duplicated()]

print("Shape of full dataset:", df.shape)

# Numeric columns (for X and Y)
numeric_cols = df.select_dtypes(include="number").columns.tolist()
print("\nNumeric columns (for X and Y):")
for c in numeric_cols:
    print(" -", c)

# Grouping columns (for p-value comparison – small number of categories)
group_cols = []
for c in df.columns:
    nunique = df[c].nunique(dropna=False)
    if 1 < nunique <= 15:
        group_cols.append(c)

print("\nGrouping columns (you can use these for p-value):")
for c in group_cols:
    print(" -", c)


# -----------------------------
# 2. Plot function (line + optional SD band + stats)
# -----------------------------
def plot_line_with_sd(x_col, y_col, show_sd, group_for_pvalue):
    """
    Plot Y vs X as a single line with optional SD band.
    Also print median, mode, and (optionally) p-value for Y.
    Uses only the first 200 valid rows.
    """
    # Columns needed
    cols = [x_col, y_col]
    if group_for_pvalue != "None":
        cols.append(group_for_pvalue)

    # Only first 200 rows after dropping NAs
    data = df[cols].dropna().head(200)

    if data.empty:
        print("No data available for this combination.")
        return

    print("\n" + "=" * 60)
    print(f"Total records in dataset: {len(df)}")
    print(f"Records used in this view (non-missing {cols}): {len(data)}")

    # ---- summary stats for Y (this view only) ----
    y_series = data[y_col]
    median_val = y_series.median()
    modes = y_series.mode()
    mode_val = modes.iloc[0] if len(modes) > 0 else np.nan

    print(f"\nSummary for '{y_col}' in this view:")
    print(f"  Median: {median_val:.2f}")
    print(f"  Mode:   {mode_val:.2f}")

    # ---- p-value, if a grouping column is chosen ----
    if group_for_pvalue != "None":
        groups = data[group_for_pvalue].unique()
        print(f"\nGroups in '{group_for_pvalue}': {groups}")

        if len(groups) == 2:
            g1 = data[data[group_for_pvalue] == groups[0]][y_col]
            g2 = data[data[group_for_pvalue] == groups[1]][y_col]

            if len(g1) > 1 and len(g2) > 1:
                _, p_val = ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
                print(f"  p-value ({groups[0]} vs {groups[1]}): {p_val:.3f}")
            else:
                print("  Not enough data in one of the groups for p-value.")
        else:
            print("  p-value only makes sense when there are exactly 2 groups.")
    else:
        print("\nNo group selected for p-value (set 'Group (for p-value)' if needed).")

    # ---- plotting (line + optional SD band) ----
    # Group by X to get mean & SD per X (in case repeated X values)
    grouped = data.groupby(x_col)[y_col]
    x_vals = grouped.mean().index.values
    y_mean = grouped.mean().values
    y_sd = grouped.std().values  # SD per X

    fig, ax = plt.subplots(figsize=(8, 5))

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

    # Extend x-limits so text isn’t cut off
    ax.set_xlim(x_vals[0], x_vals[-1] + 4 * x_offset)

    # Labels & title
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col} (first 200 records)")

    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 3. Interactive UI (for Jupyter)
# -----------------------------
def main():
    # sensible defaults
    default_x = "Day of Life" if "Day of Life" in numeric_cols else numeric_cols[0]
    default_y = "SVC flow (mL/ kg/ min)" if "SVC flow (mL/ kg/ min)" in numeric_cols else numeric_cols[0]
    default_group_for_p = "None"

    x_dropdown = Dropdown(
        options=numeric_cols,
        value=default_x,
        description="X-axis:",
        layout={"width": "400px"},
    )

    y_dropdown = Dropdown(
        options=numeric_cols,
        value=default_y,
        description="Y-axis:",
        layout={"width": "400px"},
    )

    sd_checkbox = Checkbox(
        value=False,
        description="Show standard deviation band",
    )

    group_for_p_dropdown = Dropdown(
        options=["None"] + group_cols,
        value=default_group_for_p,
        description="Group (for p-value):",
        layout={"width": "400px"},
    )

    def update_plot(x_col, y_col, show_sd, group_for_pvalue):
        plot_line_with_sd(x_col, y_col, show_sd, group_for_pvalue)

    interact(
        update_plot,
        x_col=x_dropdown,
        y_col=y_dropdown,
        show_sd=sd_checkbox,
        group_for_pvalue=group_for_p_dropdown,
    )


if __name__ == "__main__":
    # When run via `%run hemodynamics_line_sd.py` in Jupyter,
    # this will display the widgets and let you interact.
    main()

