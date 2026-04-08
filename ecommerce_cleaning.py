import pandas as pd
import numpy as np

# ===== Load =====
df = pd.read_excel(
    r"C:\Users\tamil\OneDrive\Desktop\Ecommerce_Customer_Analysis_Raw_Dataset.xlsx",
    sheet_name='Raw_Ecommerce_Data')

print("Before:", df.shape)

# ===== 1. Duplicates =====
df = df.drop_duplicates()
print("After duplicates:", df.shape)

# ===== 2. Date Fix =====
def fix_date(val):
    for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y']:
        try:
            return pd.to_datetime(val, format=fmt)
        except:
            pass
    return pd.NaT

df['Order_Date'] = df['Order_Date'].apply(fix_date)

# ===== 3. Revenue Fix =====
df['Revenue'] = df['Revenue'].astype(str).str.replace('₹','',regex=False)
df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
df = df[df['Revenue'] > 0]

# ===== 4. Null Fix =====
df['Customer_Name'].fillna('Unknown', inplace=True)
df['City'].fillna('Unknown', inplace=True)
df['Customer_Rating'].fillna(df['Customer_Rating'].median(), inplace=True)
df['Revenue'].fillna(df['Revenue'].median(), inplace=True)

# ===== 5. Text Standardize =====
df['Category'] = df['Category'].str.strip().str.title()
df['Order_Status'] = df['Order_Status'].str.strip().str.title()
df['Customer_Name'] = df['Customer_Name'].str.strip()
df['City'] = df['City'].str.strip().str.title()

# ===== 6. Invalid Remove =====
df = df[df['Quantity'] > 0]
df = df[df['Discount_Pct'] <= 100]
df['Customer_Rating'] = df['Customer_Rating'].clip(1, 5)
df = df[df['Profit'] < 500000]

# ===== 7. NumPy Analysis =====
print("\n===== NumPy Statistical Analysis =====")
revenue = df['Revenue'].values
profit = df['Profit'].values

print(f"Revenue Mean:   ₹{np.mean(revenue):,.2f}")
print(f"Revenue Median: ₹{np.median(revenue):,.2f}")
print(f"Revenue Std:    ₹{np.std(revenue):,.2f}")
print(f"Revenue 25%:    ₹{np.percentile(revenue, 25):,.2f}")
print(f"Revenue 75%:    ₹{np.percentile(revenue, 75):,.2f}")
print(f"Profit Mean:    ₹{np.mean(profit):,.2f}")
print(f"Total Revenue:  ₹{np.sum(revenue):,.2f}")
print(f"Total Profit:   ₹{np.sum(profit):,.2f}")

# ===== Save =====
df.to_csv(r"C:\Users\tamil\OneDrive\Desktop\Ecommerce_Customer_Analysis_Raw_Dataset.csv",
          index=False)
print(f"\n✅ Cleaning Done! Rows: {len(df)}")