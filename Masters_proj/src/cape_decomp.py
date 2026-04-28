"""
CAPE-aikasarjan dekompositio.

Hajottaa CAPE-aikasarjan kolmeen komponenttiin:
  - trendi (pitkän aikavälin liike)
  - kausivaihtelu (12 kuukauden jakso)
  - jäännös (loput, eli satunnaisvaihtelu)

Käyttää sekä klassista additiivista dekompositiota että robustimpaa
STL-dekompositiota (Seasonal-Trend decomposition using LOESS).

Vaatimukset:
  pip install pandas numpy matplotlib statsmodels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL


# ---------------------------------------------------------------------------
# 1. Datan lataus ja esikäsittely
# ---------------------------------------------------------------------------
df = pd.read_csv(
    "/Users/kayttaja/Desktop/Masters_proj/data/interim/Shiller_cleaned_df.csv",
    parse_dates=["date"],
)
df = df.sort_values("date").reset_index(drop=True)

# Otetaan vain rivit, joilla on CAPE-arvo, ja indeksoidaan päivämäärällä
cape = (
    df.dropna(subset=["cape"]).set_index("date")["cape"].asfreq("MS")
)  # Month Start -taajuus, jotta dekompositio toimii

# Mahdolliset puuttuvat arvot interpoloidaan lineaarisesti
cape = cape.interpolate(method="linear")

print(f"Aikasarjan pituus: {len(cape)} kuukautta")
print(f"Ajanjakso: {cape.index.min().date()} - {cape.index.max().date()}")


# ---------------------------------------------------------------------------
# 2. Klassinen additiivinen dekompositio (liukuvan keskiarvon avulla)
# ---------------------------------------------------------------------------
# period=12 tarkoittaa 12 kuukauden kausijaksoa.
classical = seasonal_decompose(cape, model="additive", period=12)


# ---------------------------------------------------------------------------
# 3. STL-dekompositio (joustavampi, kestää poikkeavia havaintoja paremmin)
# ---------------------------------------------------------------------------
# robust=True vähentää poikkeamien (esim. 1929, 2000, 2008) vaikutusta.
stl = STL(cape, period=12, robust=True).fit()


# ---------------------------------------------------------------------------
# 4. Visualisointi - klassinen dekompositio
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(cape.index, cape.values, color="steelblue")
axes[0].set_ylabel("Alkup. CAPE")
axes[0].set_title("Klassinen additiivinen dekompositio (jakso = 12 kk)")
axes[0].grid(alpha=0.3)

axes[1].plot(classical.trend.index, classical.trend.values, color="darkorange")
axes[1].set_ylabel("Trendi")
axes[1].grid(alpha=0.3)

axes[2].plot(classical.seasonal.index, classical.seasonal.values, color="seagreen")
axes[2].set_ylabel("Kausivaihtelu")
axes[2].grid(alpha=0.3)

axes[3].plot(
    classical.resid.index, classical.resid.values, color="firebrick", linewidth=0.6
)
axes[3].axhline(0, color="black", linewidth=0.6)
axes[3].set_ylabel("Jäännös")
axes[3].set_xlabel("Päivämäärä")
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# 5. Visualisointi - STL
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(cape.index, cape.values, color="steelblue")
axes[0].set_ylabel("Alkup. CAPE")
axes[0].set_title("STL-dekompositio (robust=True, jakso = 12 kk)")
axes[0].grid(alpha=0.3)

axes[1].plot(stl.trend.index, stl.trend.values, color="darkorange")
axes[1].set_ylabel("Trendi")
axes[1].grid(alpha=0.3)

axes[2].plot(stl.seasonal.index, stl.seasonal.values, color="seagreen")
axes[2].set_ylabel("Kausivaihtelu")
axes[2].grid(alpha=0.3)

axes[3].plot(stl.resid.index, stl.resid.values, color="firebrick", linewidth=0.6)
axes[3].axhline(0, color="black", linewidth=0.6)
axes[3].set_ylabel("Jäännös")
axes[3].set_xlabel("Päivämäärä")
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# 6. Yhteenvetotilastot
# ---------------------------------------------------------------------------
# Kausivaihtelun voimakkuus = 1 - Var(jäännös) / Var(jäännös + kausi)
# Lähde: Hyndman & Athanasopoulos, "Forecasting: Principles and Practice"
seasonal_strength = max(0, 1 - np.var(stl.resid) / np.var(stl.resid + stl.seasonal))
trend_strength = max(0, 1 - np.var(stl.resid) / np.var(stl.resid + stl.trend))

print("STL-dekomposition voimakkuusmittarit:")
print(f"  Trendin voimakkuus:      {trend_strength:.4f}")
print(f"  Kausivaihtelun voimakkuus: {seasonal_strength:.4f}")
print("  (Asteikko 0-1; mitä lähempänä 1, sitä voimakkaampi komponentti.)")
