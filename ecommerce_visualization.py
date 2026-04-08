import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sqlalchemy import create_engine

engine = create_engine(
    'postgresql+psycopg2://postgres:postgres123@127.0.0.1:5433/sales_db'
)

df = pd.read_sql_query('SELECT * FROM ecommerce', engine)
df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')
df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')

# ===== NumPy Analysis =====
print("===== NumPy Statistical Analysis =====")
revenue = df['Revenue'].dropna().values
profit = df['Profit'].dropna().values

print(f"Total Revenue:  ₹{np.sum(revenue)/1000000:.2f}M")
print(f"Total Profit:   ₹{np.sum(profit)/1000000:.2f}M")
print(f"Revenue Mean:   ₹{np.mean(revenue):,.2f}")
print(f"Revenue Median: ₹{np.median(revenue):,.2f}")
print(f"Revenue Std:    ₹{np.std(revenue):,.2f}")
print(f"25th Percentile:₹{np.percentile(revenue, 25):,.2f}")
print(f"75th Percentile:₹{np.percentile(revenue, 75):,.2f}")
print(f"Profit Margin:  {np.mean(profit/revenue)*100:.1f}%")

# ===== Visualization =====
plt.rcParams['figure.facecolor'] = '#0D1B2A'
plt.rcParams['axes.facecolor'] = '#1E2A3A'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

fig = plt.figure(figsize=(24, 16), facecolor='#0D1B2A')
fig.suptitle('🛒 E-COMMERCE CUSTOMER ANALYSIS DASHBOARD',
             fontsize=22, fontweight='bold', color='white', y=0.99)

gs = gridspec.GridSpec(3, 4, figure=fig,
                       hspace=0.5, wspace=0.4,
                       left=0.05, right=0.98,
                       top=0.93, bottom=0.07)

# ===== KPI Cards =====
kpis = [
    ('🛒 Total Orders', f"{len(df):,}", '#00B4D8'),
    ('💰 Total Revenue', f"₹{np.sum(revenue)/1000000:.2f}M", '#00FF99'),
    ('📈 Total Profit', f"₹{np.sum(profit)/1000000:.2f}M", '#FF6B6B'),
    ('⭐ Avg Rating', f"{df['Customer_Rating'].mean():.1f}/5", '#FFD700'),
]
for i, (title, value, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor('#162032')
    ax.text(0.5, 0.62, value, ha='center', va='center',
            fontsize=26, fontweight='bold', color=color,
            transform=ax.transAxes)
    ax.text(0.5, 0.28, title, ha='center', va='center',
            fontsize=11, color='white', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)

# ===== Category Bar Chart =====
ax2 = fig.add_subplot(gs[1, :2])
cat = df.groupby('Category')['Revenue'].sum().sort_values()/1000000
colors = ['#00B4D8','#0077B6','#48CAE4','#90E0EF','#ADE8F4','#00FF99']
bars = ax2.barh(cat.index, cat.values,
                color=colors[:len(cat)], edgecolor='#0D1B2A')
ax2.set_title('📊 Revenue by Category (M ₹)',
              fontsize=13, fontweight='bold', pad=10)
for bar, val in zip(bars, cat.values):
    ax2.text(bar.get_width()+0.05,
             bar.get_y()+bar.get_height()/2,
             f'₹{val:.2f}M', va='center',
             color='white', fontsize=9, fontweight='bold')

# ===== Payment Method Pie =====
ax3 = fig.add_subplot(gs[1, 2])
pay = df.groupby('Payment_Method')['Revenue'].sum()
colors_pie = ['#00B4D8','#FF6B6B','#00FF99','#FFD700','#FF69B4','#9C27B0']
wedges, texts, autotexts = ax3.pie(
    pay.values, labels=pay.index,
    autopct='%1.1f%%', colors=colors_pie,
    startangle=90, pctdistance=0.78)
for text in texts:
    text.set_color('white')
    text.set_fontsize(8)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(8)
ax3.set_title('💳 Payment Methods',
              fontsize=13, fontweight='bold', pad=10)

# ===== Order Status =====
ax4 = fig.add_subplot(gs[1, 3])
status = df['Order_Status'].value_counts()
colors_s = ['#00FF99','#FF6B6B','#FFD700','#00B4D8']
ax4.bar(status.index, status.values,
        color=colors_s[:len(status)], edgecolor='#0D1B2A')
ax4.set_title('📦 Order Status',
              fontsize=13, fontweight='bold', pad=10)
ax4.set_xticklabels(status.index, rotation=30, ha='right', fontsize=8)
for i, val in enumerate(status.values):
    ax4.text(i, val+5, str(val), ha='center',
             color='white', fontsize=9, fontweight='bold')

# ===== Monthly Trend =====
ax5 = fig.add_subplot(gs[2, :2])
df['Month'] = df['Order_Date'].dt.to_period('M').astype(str)
monthly = df.groupby('Month')['Revenue'].sum().sort_index()/1000000
monthly = monthly[monthly.index != 'NaT']
ax5.plot(range(len(monthly)), monthly.values,
         color='#00B4D8', linewidth=2.5,
         marker='o', markersize=5)
ax5.fill_between(range(len(monthly)),
                 monthly.values, alpha=0.2, color='#00B4D8')
step = max(1, len(monthly)//10)
ax5.set_xticks(range(0, len(monthly), step))
ax5.set_xticklabels(monthly.index[::step],
                    rotation=45, ha='right', fontsize=8)
ax5.set_title('📈 Monthly Revenue Trend (M ₹)',
              fontsize=13, fontweight='bold', pad=10)
ax5.set_ylabel('Revenue (M ₹)')

# ===== Correlation Heatmap =====
ax6 = fig.add_subplot(gs[2, 2:])
df_corr = df[['Revenue','Profit','Quantity',
              'Discount_Pct','Customer_Rating']]\
            .apply(pd.to_numeric, errors='coerce')
sns.heatmap(df_corr.corr(), annot=True, fmt='.2f',
            cmap='coolwarm', ax=ax6,
            linewidths=0.5, annot_kws={'size': 10})
ax6.set_title('🔥 Correlation Heatmap',
              fontsize=13, fontweight='bold', pad=10)

plt.savefig(
    r"C:\Users\tamil\OneDrive\Desktop\ecommerce_dashboard.png",
    dpi=150, bbox_inches='tight', facecolor='#0D1B2A')
plt.show()
print("✅ Visualization Done!")