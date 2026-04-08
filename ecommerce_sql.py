import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(
    'postgresql+psycopg2://postgres:postgres123@127.0.0.1:5433/sales_db'
)

# 1. Total KPIs
q1 = pd.read_sql_query("""
    SELECT 
        COUNT(*) AS Total_Orders,
        ROUND(SUM("Revenue")::numeric,2) AS Total_Revenue,
        ROUND(SUM("Profit")::numeric,2) AS Total_Profit,
        ROUND(AVG("Customer_Rating")::numeric,2) AS Avg_Rating
    FROM ecommerce""", engine)
print("=== TOTAL KPIs ===")
print(q1)

# 2. Category wise
q2 = pd.read_sql_query("""
    SELECT "Category",
           COUNT(*) AS Orders,
           ROUND(SUM("Revenue")::numeric,2) AS Revenue,
           ROUND(SUM("Profit")::numeric,2) AS Profit
    FROM ecommerce
    GROUP BY "Category"
    ORDER BY Revenue DESC""", engine)
print("\n=== CATEGORY WISE ===")
print(q2)

# 3. Top Customers
q3 = pd.read_sql_query("""
    SELECT "Customer_Name",
           COUNT(*) AS Orders,
           ROUND(SUM("Revenue")::numeric,2) AS Total_Revenue
    FROM ecommerce
    GROUP BY "Customer_Name"
    ORDER BY Total_Revenue DESC
    LIMIT 5""", engine)
print("\n=== TOP CUSTOMERS ===")
print(q3)

# 4. Payment Method
q4 = pd.read_sql_query("""
    SELECT "Payment_Method",
           COUNT(*) AS Orders,
           ROUND(SUM("Revenue")::numeric,2) AS Revenue
    FROM ecommerce
    GROUP BY "Payment_Method"
    ORDER BY Revenue DESC""", engine)
print("\n=== PAYMENT METHODS ===")
print(q4)

# 5. Order Status
q5 = pd.read_sql_query("""
    SELECT "Order_Status",
           COUNT(*) AS Count,
           ROUND(COUNT(*)*100.0/
           (SELECT COUNT(*) FROM ecommerce),1) AS Percentage
    FROM ecommerce
    GROUP BY "Order_Status"
    ORDER BY Count DESC""", engine)
print("\n=== ORDER STATUS ===")
print(q5)

# 6. Monthly Trend
q6 = pd.read_sql_query("""
    SELECT TO_CHAR("Order_Date"::DATE,'YYYY-MM') AS Month,
           ROUND(SUM("Revenue")::numeric,2) AS Revenue,
           COUNT(*) AS Orders
    FROM ecommerce
    WHERE "Order_Date" IS NOT NULL
    GROUP BY Month
    ORDER BY Month""", engine)
print("\n=== MONTHLY TREND ===")
print(q6)

print("\n✅ SQL Analysis Done!")