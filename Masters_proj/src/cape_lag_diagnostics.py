import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(
    "/Users/kayttaja/Desktop/Masters_proj/data/interim/Shiller_cleaned_df.csv",
    parse_dates=["date"],
)
df = df.sort_values("date").reset_index(drop=True)

# Create 12 and 24 month lag
df["cape_lag12"] = df["cape"].shift(12)
df["cape_lag24"] = df["cape"].shift(24)

# Drop na's
plot_df = df.dropna(subset=["cape", "cape_lag12", "cape_lag24"]).copy()

# Calculate correlation and regression line
corr = plot_df["cape"].corr(plot_df["cape_lag24"])
slope, intercept = np.polyfit(plot_df["cape_lag24"], plot_df["cape"], 1)

# Plot the regression
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(
    plot_df["cape_lag24"],
    plot_df["cape"],
    c=plot_df["date"].dt.year,
    cmap="viridis",
    alpha=0.6,
    s=15,
)

# 45 degree reference line
lims = [
    plot_df[["cape", "cape_lag24"]].min().min() - 1,
    plot_df[["cape", "cape_lag24"]].max().max() + 1,
]
ax.plot(lims, lims, "r--", alpha=0.7, label="45° viiteviiva (ei muutosta)")

# Fitted regression line
x_fit = np.array(lims)
ax.plot(
    x_fit,
    slope * x_fit + intercept,
    "k-",
    alpha=0.7,
    label=f"OLS-sovitus: y = {slope:.3f}x + {intercept:.3f}",
)

ax.set_xlabel("CAPE viiveellä 24 kk (t-24)", fontsize=12)
ax.set_ylabel("CAPE nyt (t)", fontsize=12)
ax.set_title(
    f"CAPE-arvon nykyhetki vs. 24 kk viive\n"
    f"Pearson-korrelaatio = {corr:.4f}, n = {len(plot_df)}",
    fontsize=13,
)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# Colorbar for the scatter plot
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Vuosi", fontsize=11)

ax.set_xlim(lims[0], lims[1])
ax.set_ylim(lims[0], lims[1])
ax.set_aspect("equal")

plt.tight_layout()
plt.show()

print(f"Korrelaatio CAPE(t) ja CAPE(t-24) välillä: {corr:.4f}")
print(f"Kulmakerroin: {slope:.4f}")
print(f"Vakiotermi: {intercept:.4f}")
print(f"Havaintojen lukumäärä: {len(plot_df)}")
