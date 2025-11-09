import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np  # for linear regression / extrapolation

# --- Page setup ---
st.set_page_config(page_title="Mai Shan Yun Inventory Dashboard", layout="wide")

st.title("Mai Shan Yun Inventory Intelligence Dashboard")
st.write("Visualize ingredient usage, purchases, and trends across months.")

# --- Load Data ---
@st.cache_data
def load_data():
    months = [
        ("May_Data_Matrix (1).xlsx", "May"),
        ("June_Data_Matrix.xlsx", "June"),
        ("July_Data_Matrix (1).xlsx", "July"),
        ("August_Data_Matrix (1).xlsx", "August"),
        ("September_Data_Matrix.xlsx", "September"),
        ("October_Data_Matrix_20251103_214000.xlsx", "October"),
    ]

    frames = []
    for file, month in months:
        try:
            df = pd.read_excel(file)
            df["Month"] = month
            frames.append(df)
        except FileNotFoundError:
            st.warning(f"⚠️ Could not find file: {file}")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

data = load_data()

@st.cache_data
def load_ingredient_usage():
    return pd.read_csv("MSY Data - Ingredient.csv")

ingredient_usage_raw = load_ingredient_usage()

# --- Clean data ---
# remove metadata columns
data = data.drop(columns=["source_page", "source_table"], errors="ignore")

# strip whitespace from all column names to avoid issues like "Count " vs "Count"
data.columns = data.columns.map(str).str.strip()

pretty_names = {
    "Group": "Menu Type",
    "Count": "Items Sold",
    "Amount": "Total Sales ($)",
    "Month": "Month",
}
data = data.rename(columns=pretty_names)

# ensure Items Sold is numeric
if "Items Sold" in data.columns:
    data["Items Sold"] = pd.to_numeric(data["Items Sold"], errors="coerce")

if "Menu Type" in data.columns:
    data["Menu Type"] = data["Menu Type"].fillna("Unspecified Menu")

# basic safety check
if data.empty:
    st.error("❌ No data loaded. Check your Excel files.")
    st.stop()

st.markdown("---")

# ========== TOP KPI CARDS (Custom) ==========

# 1) Total Sales Revenue (money earned)
total_sales_revenue = None
if "Total Sales ($)" in data.columns:
    sales_clean = (
        data["Total Sales ($)"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    sales_num = pd.to_numeric(sales_clean, errors="coerce")
    total_sales_revenue = float(sales_num.sum())
    revenue_text = f"${total_sales_revenue:,.0f}" if not pd.isna(total_sales_revenue) else "N/A"
else:
    revenue_text = "N/A"

# 2) Top 3 Ingredients Used (from ingredient usage CSV)
top_3_ingredients_text = "N/A"
usage_summary_kpi = pd.DataFrame()

if ingredient_usage_raw is not None and not ingredient_usage_raw.empty:
    df_ing = ingredient_usage_raw.rename(columns={"Item name": "Menu Item"})
    long_ing = df_ing.melt(
        id_vars="Menu Item",
        var_name="Ingredient",
        value_name="Amount per Serving"
    )
    long_ing = long_ing.dropna(subset=["Amount per Serving"])
    long_ing = long_ing[long_ing["Amount per Serving"] > 0]

    usage_summary_kpi = (
        long_ing
        .groupby("Ingredient", as_index=False)["Amount per Serving"]
        .sum()
        .sort_values("Amount per Serving", ascending=False)
    )

    def clean_ing_name(s):
        s = str(s)
        return s.split("(")[0].strip() if "(" in s else s.strip()

    top_3_ing = usage_summary_kpi.head(3)["Ingredient"].apply(clean_ing_name).tolist()
    if top_3_ing:
        top_3_ingredients_text = ", ".join(top_3_ing)

# 3) Estimated ingredient cost (money spent on ingredients)
# 3) Estimated ingredient cost (money spent on ingredients)
ingredient_costs = {
    "braised beef used (g)": 0.02,
    "Braised Chicken(g)": 0.015,
    "Braised Pork(g)": 0.018,
    "Egg(count)": 0.25,
    "Rice(g)": 0.004,
    "Ramen (count)": 0.40,
    "Rice Noodles(g)": 0.005,
    "chicken thigh (pcs)": 0.80,
    "Chicken Wings (pcs)": 0.50,
    "flour (g)": 0.002,
    "Pickle Cabbage": 0.004,
    "Green Onion": 0.004,
    "Cilantro": 0.003,
    "White onion": 0.004,
    "Peas(g)": 0.004,
    "Carrot(g)": 0.003,
    "Boychoy(g)": 0.005,
    "Tapioca Starch": 0.003,
}

total_ingredient_cost = None

# Need ingredient usage + how many items were sold
if (
    ingredient_usage_raw is not None
    and not ingredient_usage_raw.empty
    and "Items Sold" in data.columns
):
    try:
        # 1) Get per-dish ingredient usage (average across menu items)
        df_ing_cost = ingredient_usage_raw.rename(columns={"Item name": "Menu Item"}).fillna(0)

        if "Menu Item" in df_ing_cost.columns:
            df_ing_cost = df_ing_cost.set_index("Menu Item")

            # avg_per_dish: for each ingredient, average amount used per dish
            avg_per_dish = df_ing_cost.mean(axis=0)  # one number per ingredient

            # 2) Total dishes sold in your historical period (May–Oct)
            total_dishes_sold = pd.to_numeric(data["Items Sold"], errors="coerce").fillna(0).sum()

            # 3) Compute total cost = sum over all ingredients
            total_ingredient_cost = 0.0
            for ing_name, per_dish_amount in avg_per_dish.items():
                cost_per_unit = ingredient_costs.get(ing_name, 0.0)
                # per_dish_amount (e.g. grams per bowl) * number of bowls * cost per gram
                total_ingredient_cost += float(per_dish_amount) * float(total_dishes_sold) * cost_per_unit

    except Exception as e:
        st.warning(f"Could not compute estimated ingredient cost: {e}")
        total_ingredient_cost = None

ingredient_cost_text = (
    f"${total_ingredient_cost:,.0f}"
    if total_ingredient_cost is not None and not pd.isna(total_ingredient_cost)
    else "N/A"
)


# 4) Estimated gross profit
profit_text = "N/A"
if (
    total_sales_revenue is not None and not pd.isna(total_sales_revenue)
    and total_ingredient_cost is not None and not pd.isna(total_ingredient_cost)
):
    est_profit = total_sales_revenue - total_ingredient_cost
    profit_text = f"${est_profit:,.0f}"

# 5) Estimated Top 3 Menu Items using ingredient consumption
top_3_menu_items_text = "N/A"
if ingredient_usage_raw is not None and not ingredient_usage_raw.empty:
    ingredient_df = ingredient_usage_raw.rename(columns={"Item name": "Menu Item"}).fillna(0)
    if "Menu Item" in ingredient_df.columns:
        ingredient_df = ingredient_df.set_index("Menu Item")
        total_usage = ingredient_df.sum(axis=0)
        estimates = {}
        for menu_item, row in ingredient_df.iterrows():
            ests = [
                total_usage.get(ing, 0) / val
                for ing, val in row.items()
                if val > 0 and total_usage.get(ing, 0) > 0
            ]
            if ests:
                estimates[menu_item] = sum(ests) / len(ests)
        if estimates:
            top_estimates = sorted(estimates.items(), key=lambda x: x[1], reverse=True)[:3]
            top_3_menu_items_text = ", ".join([n for n, _ in top_estimates])
            st.session_state["top_estimates_df"] = pd.DataFrame(
                top_estimates, columns=["Menu Item", "Estimated Servings"]
            )

# --- Display KPI rows ---

# Row 1: Financial KPIs
f1, f2, f3 = st.columns(3)
f1.metric("💰 Total Sales Revenue", revenue_text)
f2.metric("🧾 Est. Ingredient Cost", ingredient_cost_text)
f3.metric("💸 Est. Gross Profit", profit_text)

# Row 2: Top items summary
t1, t2 = st.columns(2)
t1.markdown(
    f"""
    <div style="text-align:center">
        <p style="font-size:18px; font-weight:600; margin-bottom:4px;">🥕 Top 3 Ingredients Used</p>
        <p style="font-size:16px; color: white;">{top_3_ingredients_text}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
t2.markdown(
    f"""
    <div style="text-align:center">
        <p style="font-size:18px; font-weight:600; margin-bottom:4px;">🍽️ Top 3 Menu Items (Estimated)</p>
        <p style="font-size:16px; color: white;">{top_3_menu_items_text}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ========== SIDEBAR FILTERS ==========
st.sidebar.header("Filters")
month_col = "Month" if "Month" in data.columns else None
if month_col:
    months = sorted(data[month_col].unique())
    selected_months = st.sidebar.multiselect("Select month(s)", options=months, default=months)
    filtered_data = data[data[month_col].isin(selected_months)]
else:
    filtered_data = data.copy()

ingredient_col = "Menu Type"
if ingredient_col in filtered_data.columns:
    ing_opts = sorted(filtered_data[ingredient_col].unique())
    sel_ing = st.sidebar.selectbox("Focus on one menu type (optional)", ["All"] + ing_opts)
    if sel_ing != "All":
        filtered_data = filtered_data[filtered_data[ingredient_col] == sel_ing]

# ========== MAIN TABS ==========
tab_overview, tab_ing, tab_forecast, tab_ship = st.tabs(
    ["📊 Overview", "🥕 Ingredient Usage", "📈 Forecast", "🚚 Shipping"]
)
    

# ===== OVERVIEW TAB =====
with tab_overview:
    st.subheader("Overall Trends")
    st.write("Sample of current filtered data:")
    st.dataframe(filtered_data.head(20), use_container_width=True)

    y_options = []
    if "Items Sold" in filtered_data.columns:
        y_options.append("Items Sold")
    if "Total Sales ($)" in filtered_data.columns:
        y_options.append("Total Sales ($)")

    if month_col and y_options:
        y_col = st.selectbox(
            "Select metric to plot over time",
            options=y_options,
            index=0,
            key="overview_y_col",
        )

        if y_col == "Total Sales ($)":
            sales_clean_plot = (
                filtered_data["Total Sales ($)"]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            sales_num_plot = pd.to_numeric(sales_clean_plot, errors="coerce")
            temp = filtered_data.copy()
            temp["_y"] = sales_num_plot
            trend_df = (
                temp.groupby(month_col, as_index=False)["_y"]
                .sum()
                .sort_values(month_col)
            )
            fig = px.line(
                trend_df,
                x=month_col,
                y="_y",
                markers=True,
                labels={"_y": "Total Sales ($)"},
                title="Total Sales ($) by Month",
            )
        else:
            trend_df = (
                filtered_data.groupby(month_col, as_index=False)[y_col]
                .sum()
                .sort_values(month_col)
            )
            fig = px.line(
                trend_df,
                x=month_col,
                y=y_col,
                markers=True,
                title=f"{y_col} by Month",
            )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add a 'Month' column and at least one metric to see trend charts here.")

    if "top_estimates_df" in st.session_state:
        st.markdown("### 🍽️ Estimated Top Menu Items by Ingredient Usage")
        fig_est = px.bar(
            st.session_state["top_estimates_df"],
            x="Menu Item",
            y="Estimated Servings",
            title="Estimated Top Menu Items (based on ingredient usage)",
        )
        st.plotly_chart(fig_est, use_container_width=True)

# ===== INGREDIENT USAGE TAB =====
with tab_ing:
    st.subheader("🥕 Ingredient Usage Summary")
    st.caption("Simple view of how much each ingredient is used across menu items.")

    if ingredient_usage_raw is None or ingredient_usage_raw.empty:
        st.warning("No ingredient-usage data found.")
    else:
        df = ingredient_usage_raw.rename(columns={"Item name": "Menu Item"})
        long_df = df.melt(
            id_vars="Menu Item",
            var_name="Ingredient",
            value_name="Amount per Serving"
        )
        long_df = long_df.dropna(subset=["Amount per Serving"])
        long_df = long_df[long_df["Amount per Serving"] > 0]

        usage_summary = (
            long_df.groupby("Ingredient", as_index=False)
            .agg(
                Total_Amount=("Amount per Serving", "sum"),
                Menu_Items_Count=("Menu Item", "nunique")
            )
            .sort_values("Total_Amount", ascending=False)
        )

        def split_name_and_unit(s):
            s = str(s)
            if "(" in s and s.endswith(")"):
                base, unit = s[:-1].split("(", 1)
                return base.strip(), unit.strip()
            return s.strip(), ""

        unit_overrides = {
            "Pickle Cabbage": "g",
            "Green Onion": "g",
            "Cilantro": "g",
            "White onion": "g",
            "Tapioca Starch": "g",
        }

        display_rows = []
        for _, r in usage_summary.iterrows():
            base, unit = split_name_and_unit(r["Ingredient"])
            final_unit = unit or unit_overrides.get(base, "")
            formatted = f"{r['Total_Amount']:.2f} {final_unit}" if final_unit else f"{r['Total_Amount']:.2f}"
            display_rows.append({
                "Ingredient": base,
                "Amount Used per Serving (total)": formatted,
                "Number of Menu Items Using Ingredient": int(r["Menu_Items_Count"]),
            })

        st.markdown("### 📋 Ingredient Usage Table")
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True)

        st.markdown("### 🔝 Top Ingredients by Usage")
        top_n = st.slider("Show top N ingredients:", 5, 20, 10)
        top_usage = usage_summary.head(top_n)
        fig_top_ing = px.bar(
            top_usage,
            x="Ingredient",
            y="Total_Amount",
            title="Top Ingredients by Total Usage",
            labels={"Total_Amount": "Total Amount per Serving"},
        )
        st.plotly_chart(fig_top_ing, use_container_width=True)

        st.markdown("### 🔎 Check a Specific Ingredient")
        selected_ing = st.selectbox(
            "Select an ingredient:",
            sorted(usage_summary["Ingredient"].unique())
        )
        st.dataframe(
            long_df[long_df["Ingredient"] == selected_ing][["Menu Item", "Amount per Serving"]],
            use_container_width=True,
            hide_index=True
        )

# ===== FORECAST TAB =====
with tab_forecast:
    st.subheader("📈 Forecast: Sales & Ingredient Needs")
    st.caption("Linear extrapolation based on May–October trends to estimate future demand.")

    month_order = ["May", "June", "July", "August", "September", "October"]
    future_month_labels = ["November (forecast)", "December (forecast)", "January (forecast)"]

    # ---- Sales forecast ----
    forecast_sales_df = None

    if "Items Sold" in data.columns and "Month" in data.columns:
        monthly_sales = (
            data.groupby("Month", as_index=False)["Items Sold"]
            .sum()
        )

        monthly_sales["MonthOrder"] = monthly_sales["Month"].apply(
            lambda m: month_order.index(m) + 1 if m in month_order else np.nan
        )
        monthly_sales = monthly_sales.dropna(subset=["MonthOrder"])
        monthly_sales["MonthOrder"] = monthly_sales["MonthOrder"].astype(int)

        if len(monthly_sales) >= 2:
            x = monthly_sales["MonthOrder"].astype(float).values
            y = monthly_sales["Items Sold"].astype(float).values

            # extra safety: drop any NaNs in x or y
            mask = ~np.isnan(x) & ~np.isnan(y)
            x = x[mask]
            y = y[mask]

            if len(x) >= 2:
                m, b = np.polyfit(x, y, 1)  # linear fit

                future_orders = np.arange(len(month_order) + 1, len(month_order) + 4, dtype=float)
                future_items = m * future_orders + b

                forecast_sales_df = pd.DataFrame({
                    "Month": future_month_labels,
                    "Items Sold (forecast)": future_items
                })

                st.markdown("### 📊 Items Sold: Historical vs Forecast")
                combined_sales = pd.concat([
                    monthly_sales[["Month", "Items Sold"]].assign(Type="Historical"),
                    forecast_sales_df.rename(columns={"Items Sold (forecast)": "Items Sold"}).assign(Type="Forecast")
                ])

                fig_sales = px.line(
                    combined_sales,
                    x="Month",
                    y="Items Sold",
                    color="Type",
                    markers=True,
                    title="Items Sold by Month (Historical vs Forecast)"
                )
                st.plotly_chart(fig_sales, use_container_width=True)

                st.dataframe(forecast_sales_df, use_container_width=True)
            else:
                st.info("Not enough valid numeric sales points to run a forecast.")
        else:
            st.info("Not enough months with sales data to run a forecast.")
    else:
        st.info("Items Sold or Month column missing; cannot compute sales forecast.")

    st.markdown("---")

    # ---- Revenue forecast ----
    if "Total Sales ($)" in data.columns and "Month" in data.columns:
        rev_clean = (
            data["Total Sales ($)"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        data["_rev_num"] = pd.to_numeric(rev_clean, errors="coerce")

        monthly_rev = (
            data.groupby("Month", as_index=False)["_rev_num"]
            .sum()
            .rename(columns={"_rev_num": "Revenue"})
        )
        monthly_rev["MonthOrder"] = monthly_rev["Month"].apply(
            lambda m: month_order.index(m) + 1 if m in month_order else np.nan
        )
        monthly_rev = monthly_rev.dropna(subset=["MonthOrder"])
        monthly_rev["MonthOrder"] = monthly_rev["MonthOrder"].astype(int)

        if len(monthly_rev) >= 2:
            xr = monthly_rev["MonthOrder"].astype(float).values
            yr = monthly_rev["Revenue"].astype(float).values

            mask_r = ~np.isnan(xr) & ~np.isnan(yr)
            xr = xr[mask_r]
            yr = yr[mask_r]

            if len(xr) >= 2:
                mr, br = np.polyfit(xr, yr, 1)

                future_orders_r = np.arange(len(month_order) + 1, len(month_order) + 4, dtype=float)
                future_rev = mr * future_orders_r + br

                forecast_rev_df = pd.DataFrame({
                    "Month": future_month_labels,
                    "Revenue (forecast)": future_rev
                })

                st.markdown("### 💵 Revenue: Historical vs Forecast")
                combined_rev = pd.concat([
                    monthly_rev[["Month", "Revenue"]].assign(Type="Historical"),
                    forecast_rev_df.rename(columns={"Revenue (forecast)": "Revenue"}).assign(Type="Forecast")
                ])

                fig_rev = px.line(
                    combined_rev,
                    x="Month",
                    y="Revenue",
                    color="Type",
                    markers=True,
                    title="Revenue by Month (Historical vs Forecast)"
                )
                st.plotly_chart(fig_rev, use_container_width=True)

                st.dataframe(forecast_rev_df, use_container_width=True)
            else:
                st.info("Not enough valid numeric revenue points to run a forecast.")
        else:
            st.info("Not enough months with revenue data to run a forecast.")
    else:
        st.info("Total Sales ($) or Month column missing; cannot compute revenue forecast.")

    st.markdown("---")

    # ---- Ingredient needs forecast ----
    st.subheader("🥕 Forecasted Ingredient Needs")

    if (
        ingredient_usage_raw is not None
        and not ingredient_usage_raw.empty
        and "Items Sold" in data.columns
        and forecast_sales_df is not None
    ):
        df_ing_usage = ingredient_usage_raw.rename(columns={"Item name": "Menu Item"}).fillna(0)
        if "Menu Item" in df_ing_usage.columns:
            df_ing_usage = df_ing_usage.set_index("Menu Item")
            # Average amount of each ingredient per dish (across recipes)
            avg_per_dish = df_ing_usage.mean(axis=0)  # per ingredient

            records = []
            for month_label, items_pred in zip(
                forecast_sales_df["Month"], forecast_sales_df["Items Sold (forecast)"]
            ):
                items_pred_val = max(float(items_pred), 0.0)
                for ingredient_name, per_dish_amount in avg_per_dish.items():
                    total_needed = float(per_dish_amount) * items_pred_val
                    records.append({
                        "Month": month_label,
                        "Ingredient": ingredient_name,
                        "Predicted Amount Needed": total_needed
                    })
            ing_forecast_df = pd.DataFrame(records)

            # Dropdown to inspect one ingredient
            st.markdown("#### View forecast for a specific ingredient")
            ing_choices = sorted(ing_forecast_df["Ingredient"].unique().tolist())
            selected_ing_f = st.selectbox("Choose an ingredient:", ing_choices)

            ing_selected_df = ing_forecast_df[ing_forecast_df["Ingredient"] == selected_ing_f]
            fig_ing = px.bar(
                ing_selected_df,
                x="Month",
                y="Predicted Amount Needed",
                title=f"Predicted {selected_ing_f} Needed (Next 3 Months)",
            )
            st.plotly_chart(fig_ing, use_container_width=True)

            # Top 5 ingredients by total future need
            st.markdown("#### Top 5 Ingredients by Predicted Total Usage (Next 3 Months)")
            top_ing_future = (
                ing_forecast_df.groupby("Ingredient", as_index=False)["Predicted Amount Needed"]
                .sum()
                .sort_values("Predicted Amount Needed", ascending=False)
                .head(5)
            )
            st.dataframe(top_ing_future, use_container_width=True)
        else:
            st.info("Ingredient usage file does not contain 'Item name' / 'Menu Item' structure.")
    else:
        st.info("Ingredient usage data, sales data, or sales forecast missing; cannot forecast ingredient needs.")

# ===== SHIPPING TAB =====
with tab_ship:
    st.subheader("🚚 Shipment Forecasting")
    st.caption(
        "Forecast how much of each ingredient will arrive over the next few months "
        "based on your shipment plan CSV."
    )

    # Upload the shipment plan CSV
    ship_file = st.file_uploader(
        "Upload Shipment Plan CSV",
        type=["csv"],
        key="shipment_forecast_file"
    )

    if ship_file is None:
        st.info("👆 Upload your shipment CSV (with columns: Ingredient, Quantity per shipment, Unit of shipment, Number of shipments, frequency).")
    else:
        try:
            df = pd.read_csv(ship_file)
            df.columns = [c.strip() for c in df.columns]  # remove any stray spaces
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()

        st.markdown("#### Preview – Shipment Data")
        st.dataframe(df.head(), use_container_width=True)

        # Ensure required columns exist
        required_cols = {"Ingredient", "Quantity per shipment", "Unit of shipment", "Number of shipments", "frequency"}
        missing = required_cols - set(df.columns)
        if missing:
            st.error(f"CSV missing these columns: {', '.join(missing)}")
        else:
            # Clean numeric data
            df["Quantity per shipment"] = pd.to_numeric(df["Quantity per shipment"], errors="coerce")
            df["Number of shipments"] = pd.to_numeric(df["Number of shipments"], errors="coerce")
            df["frequency"] = df["frequency"].astype(str).str.strip().str.lower()

            # Map frequency text to shipments per month
            freq_map = {
                "weekly": 4,
                "biweekly": 2,
                "bi-weekly": 2,
                "bi weekly": 2,
                "monthly": 1,
            }
            df["Shipments per Month"] = df["frequency"].map(freq_map).fillna(0)

            # Choose forecast horizon
            horizon = st.slider("Forecast Horizon (months)", 1, 12, 3)

            # Compute inbound and forecast quantities
            df["Monthly Inbound Quantity"] = df["Quantity per shipment"] * df["Shipments per Month"]
            df["Forecasted Quantity (Next Period)"] = df["Monthly Inbound Quantity"] * horizon

            # Rename columns for display
            display_df = df.rename(
                columns={
                    "Ingredient": "Ingredient Name",
                    "Unit of shipment": "Unit",
                    "frequency": "Shipment Frequency",
                    "Monthly Inbound Quantity": "Avg. Monthly Quantity",
                    "Forecasted Quantity (Next Period)": f"Forecasted Total (Next {horizon} Months)",
                }
            )[
                [
                    "Ingredient Name",
                    "Unit",
                    "Shipment Frequency",
                    "Avg. Monthly Quantity",
                    f"Forecasted Total (Next {horizon} Months)",
                ]
            ]

            st.markdown(f"### 📦 Forecasted Inbound Quantities (Next {horizon} Months)")
            st.dataframe(display_df, use_container_width=True)

            # Bar chart of forecast
            fig = px.bar(
                display_df,
                x="Ingredient Name",
                y=f"Forecasted Total (Next {horizon} Months)",
                color="Shipment Frequency",
                title=f"Forecasted Inbound Quantity per Ingredient (Next {horizon} Months)",
                labels={f"Forecasted Total (Next {horizon} Months)": "Forecasted Total Quantity"},
            )
            fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            st.success("✅ Shipment forecast generated successfully!")
