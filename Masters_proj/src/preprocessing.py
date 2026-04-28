import pandas as pd

df_raw = pd.read_excel("/Users/kayttaja/Desktop/Masters_proj/data/raw/Shiller_dataset.xls", header=7, sheet_name="Data")

print(df_raw.head())
print(df_raw.columns)

# Drop columns
df_raw = df_raw.drop(columns=['Unnamed: 13', 'Unnamed: 15'])

# Let's rename some columns
df_raw = df_raw.rename(columns={'S&P Comp. P': 'sp500_price',
                                'Date': 'date',
                                'Dividend D': 'dividend',
                                'Earnings E': 'earnings',
                                '  Consumer Price Index CPI': 'cpi',
                                'Date   Fraction': 'date_fraction',
                                'Long Interest Rate GS10': 'gs10',
                                'Real Price': 'real_price',
                                'Real Dividend': 'real_dividend',
                                'Real Total Return Price': 'real_total_return_price',
                                'Real Earnings': 'real_earnings',
                                'Real TR Scaled Earnings': 'real_tr_scaled_earnings',
                                'Cyclically Adjusted Price Earnings Ratio P/E10 or CAPE': 'cape',
                                'Cyclically  Adjusted Total Return Price Earnings Ratio TR P/E10 or TR CAPE': 'tr_cape',
                                'Excess CAPE Yield': 'ecy',
                                'Monthly Total Bond Returns': 'monthly_total_bond_returns',
                                'Real Total Bond Returns': 'real_total_bond_returns',
                                '10 Year Annualized Stock Real Return': 'ten_year_annualized_stock_real_return',
                                '10 Year Annualized Bonds  Real Return': 'ten_year_annualized_bonds_real_return',
                                'Real 10 Year Excess Annualized  Returns': 'real_10_year_excess_annualized_returns'})

print(df_raw.columns)

# date column to pd datetime
df_raw['date'] = df_raw['date'].astype(float)
df_raw['date'] = df_raw['date'] * 100
print(df_raw['date'].head())

df_raw["date"] = pd.to_datetime(
    df_raw["date"].astype(int).astype(str),
    format="%Y%m"
)
print(df_raw["date"].head(15))

print(df_raw.dtypes)

# Export to CSV
df_raw.to_csv("/Users/kayttaja/Desktop/Masters_proj/data/interim/Shiller_cleaned_df.csv", index=False)