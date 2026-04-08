import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

engine = create_engine(
    'postgresql+psycopg2://postgres:postgres123@127.0.0.1:5433/sales_db'
)

df = pd.read_sql_query('SELECT * FROM ecommerce', engine)
df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Discount_Pct'] = pd.to_numeric(df['Discount_Pct'], errors='coerce')
df = df.dropna(subset=['Revenue','Profit','Quantity','Discount_Pct'])

plt.rcParams['figure.facecolor'] = '#0D1B2A'
plt.rcParams['axes.facecolor'] = '#1E2A3A'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

fig = plt.figure(figsize=(22, 14), facecolor='#0D1B2A')
fig.suptitle('🤖 E-COMMERCE ML ANALYSIS DASHBOARD',
             fontsize=20, fontweight='bold', color='white', y=0.99)
gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.5, wspace=0.4,
                       left=0.06, right=0.98,
                       top=0.93, bottom=0.08)

# ===== 1. RFM Analysis =====
print("===== RFM Analysis =====")
df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
snapshot = df['Order_Date'].max()

rfm = df.groupby('Customer_ID').agg(
    Recency=('Order_Date', lambda x: (snapshot-x.max()).days),
    Frequency=('Order_ID', 'count'),
    Monetary=('Revenue', 'sum')
).reset_index()

rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])
rfm['RFM_Score'] = rfm['R_Score'].astype(int) + \
                   rfm['F_Score'].astype(int) + \
                   rfm['M_Score'].astype(int)

def segment(score):
    if score >= 10: return '💎 VIP'
    elif score >= 7: return '🥇 Premium'
    elif score >= 5: return '🥈 Regular'
    else: return '🥉 Occasional'

rfm['Segment'] = rfm['RFM_Score'].apply(segment)
print(rfm['Segment'].value_counts())

# RFM Scatter
ax1 = fig.add_subplot(gs[0, 0])
colors_map = {'💎 VIP': '#FFD700', '🥇 Premium': '#00B4D8',
              '🥈 Regular': '#00FF99', '🥉 Occasional': '#FF6B6B'}
for seg, color in colors_map.items():
    mask = rfm['Segment'] == seg
    ax1.scatter(rfm[mask]['Frequency'], rfm[mask]['Monetary']/1000,
                c=color, label=seg, alpha=0.7, s=40)
ax1.set_title('💎 RFM Customer Segments',
              fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Frequency (Orders)')
ax1.set_ylabel('Monetary (₹K)')
ax1.legend(fontsize=7, loc='upper left')

# ===== 2. KMeans Segmentation =====
print("\n===== KMeans Clustering =====")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    rfm[['Recency','Frequency','Monetary']])
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_names = {0:'🛍️ Bargain Hunter', 1:'💎 VIP Buyer',
                 2:'🔄 Regular', 3:'😴 Inactive'}
rfm['Customer_Type'] = rfm['Cluster'].map(cluster_names)
print(rfm['Customer_Type'].value_counts())

ax2 = fig.add_subplot(gs[0, 1])
cluster_colors = ['#00B4D8','#FFD700','#00FF99','#FF6B6B']
for i, (name, color) in enumerate(zip(cluster_names.values(),
                                       cluster_colors)):
    mask = rfm['Cluster'] == i
    ax2.scatter(rfm[mask]['Recency'],
                rfm[mask]['Monetary']/1000,
                c=color, label=name, alpha=0.7, s=40)
ax2.set_title('🎯 KMeans Clustering',
              fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Recency (Days)')
ax2.set_ylabel('Monetary (₹K)')
ax2.legend(fontsize=7, loc='upper right')

# ===== 3. Linear Regression =====
print("\n===== Linear Regression =====")
X = df[['Quantity','Discount_Pct','Unit_Price']]
y = df['Revenue']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R² Score: {r2:.3f}")
print(f"RMSE: ₹{rmse:,.2f}")

ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(y_test[:150], y_pred[:150],
            color='#00B4D8', alpha=0.6, s=30)
ax3.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', linewidth=2)
ax3.set_title(f'📈 Revenue Prediction\nR²={r2:.3f} | RMSE=₹{rmse:,.0f}',
              fontsize=12, fontweight='bold', pad=10)
ax3.set_xlabel('Actual Revenue')
ax3.set_ylabel('Predicted Revenue')

# ===== 4. Customer Type Distribution =====
ax4 = fig.add_subplot(gs[1, 0])
seg_counts = rfm['Segment'].value_counts()
colors_bar = ['#FFD700','#00B4D8','#00FF99','#FF6B6B']
bars = ax4.bar(seg_counts.index, seg_counts.values,
               color=colors_bar[:len(seg_counts)],
               edgecolor='#0D1B2A')
ax4.set_title('👥 Customer Segments',
              fontsize=12, fontweight='bold', pad=10)
ax4.set_xticklabels(seg_counts.index, rotation=20, ha='right', fontsize=8)
for bar, val in zip(bars, seg_counts.values):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             str(val), ha='center', color='white',
             fontsize=9, fontweight='bold')

# ===== 5. Revenue Forecast =====
ax5 = fig.add_subplot(gs[1, 1])
df['Month'] = df['Order_Date'].dt.to_period('M').astype(str)
monthly = df.groupby('Month')['Revenue'].sum().sort_index()/1000000
monthly = monthly[monthly.index != 'NaT']
X_time = np.array(range(len(monthly))).reshape(-1, 1)
y_time = monthly.values
model_t = LinearRegression()
model_t.fit(X_time, y_time)
future = np.array(range(len(monthly), len(monthly)+6)).reshape(-1,1)
forecast = model_t.predict(future)
ax5.plot(range(len(monthly)), y_time,
         color='#00B4D8', linewidth=2, marker='o',
         markersize=4, label='Actual')
ax5.plot(range(len(monthly), len(monthly)+6),
         forecast, color='#FF6B6B', linewidth=2,
         linestyle='--', marker='s',
         markersize=4, label='Forecast')
ax5.fill_between(range(len(monthly), len(monthly)+6),
                 forecast, alpha=0.2, color='#FF6B6B')
ax5.set_title('🔮 Revenue Forecast (6 months)',
              fontsize=12, fontweight='bold', pad=10)
ax5.set_ylabel('Revenue (M ₹)')
ax5.legend(fontsize=9)

# ===== 6. Feature Importance =====
ax6 = fig.add_subplot(gs[1, 2])
features = ['Quantity', 'Discount_Pct', 'Unit_Price']
importance = np.abs(model.coef_)
colors_f = ['#00B4D8','#FF6B6B','#00FF99']
bars = ax6.barh(features, importance,
                color=colors_f, edgecolor='#0D1B2A')
ax6.set_title('🔍 Feature Importance',
              fontsize=12, fontweight='bold', pad=10)
for bar, val in zip(bars, importance):
    ax6.text(bar.get_width()+0.5,
             bar.get_y()+bar.get_height()/2,
             f'{val:.1f}', va='center',
             color='white', fontsize=9, fontweight='bold')

plt.savefig(
    r"C:\Users\tamil\OneDrive\Desktop\ecommerce_ml.png",
    dpi=150, bbox_inches='tight', facecolor='#0D1B2A')
plt.show()
print("\n✅ ML Analysis Done!")