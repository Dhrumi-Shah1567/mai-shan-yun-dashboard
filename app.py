import streamlit as st
import pandas as pd
import plotly.express as px


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
        combined = pd.concat(frames, ignore_index=True)
        return combined
    else:
        return pd.DataFrame()  # empty fallback

data = load_data()

# ========== BASIC SAFETY CHECK ==========
if data is None or data.empty:
    st.error("❌ No data loaded. Check your Excel files and load_data() function.")
    st.stop()

# ========== TOP BAR: TITLE + DESCRIPTION ==========
st.title("🥢 Mai Shan Yun Inventory Intelligence Dashboard")
st.caption("Track ingredient usage, purchases, and operations to reduce waste and avoid shortages.")

st.markdown("---")

# ========== TOP KPI CARDS ==========
# 👉 tweak column names here once you know them
total_rows = len(data)
num_columns = len(data.columns)

# Try to guess some useful metrics; you can change these if you know exact column names
ingredient_col = "Ingredient" if "Ingredient" in data.columns else data.columns[0]
unique_ingredients = data[ingredient_col].nunique()

# If you have a cost column, use it; otherwise set to 0
cost_col_candidates = [c for c in data.columns if "cost" in c.lower() or "price" in c.lower()]
if cost_col_candidates:
    cost_col = cost_col_candidates[0]
    total_cost = data[cost_col].sum()
else:
    cost_col = None
    total_cost = 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", f"{total_rows:,}")
col2.metric("Unique Ingredients", f"{unique_ingredients:,}")
if cost_col:
    col3.metric("Total Spend", f"${total_cost:,.0f}")
else:
    col3.metric("Total Spend", "N/A")

st.markdown("---")

# ========== SIDEBAR FILTERS ==========
st.sidebar.header("Filters")

# Month filter (uses your Month column from load_data)
month_col = "Month" if "Month" in data.columns else None
if month_col:
    months_available = sorted(data[month_col].unique().tolist())
    selected_months = st.sidebar.multiselect(
        "Select month(s)",
        options=months_available,
        default=months_available,
    )
    filtered_data = data[data[month_col].isin(selected_months)]
else:
    st.sidebar.warning("No 'Month' column found. Using all data.")
    filtered_data = data.copy()

# Ingredient filter (optional)
if ingredient_col in filtered_data.columns:
    ingredients_available = sorted(filtered_data[ingredient_col].unique().tolist())
    selected_ingredient = st.sidebar.selectbox(
        "Focus on one ingredient (optional)",
        options=["All"] + ingredients_available,
    )
    if selected_ingredient != "All":
        filtered_data = filtered_data[filtered_data[ingredient_col] == selected_ingredient]

# ========== MAIN TABS ==========
tab_overview, tab_ingredients, tab_forecast, tab_shipping = st.tabs(
    ["📊 Overview", "🥕 Ingredients", "📈 Forecast (coming soon)", "🚚 Shipping (coming soon)"]
)

# ===== TAB 1: OVERVIEW =====
with tab_overview:
    st.subheader("Overall Trends")

    # Show a quick sample of the filtered data
    st.write("Sample of current filtered data:")
    st.dataframe(filtered_data.head(20), use_container_width=True)

    # Try a simple numeric-over-time chart if possible
    numeric_cols = filtered_data.select_dtypes(include="number").columns
    if month_col and len(numeric_cols) > 0:
        y_col = st.selectbox(
            "Select numeric column to plot over time",
            options=list(numeric_cols),
            index=0,
            key="overview_y_col",
        )
        trend_df = (
            filtered_data.groupby(month_col, as_index=False)[y_col]
            .sum()
            .sort_values(month_col)
        )
        st.markdown(f"**{y_col} by {month_col}**")
        fig_overview = px.line(trend_df, x=month_col, y=y_col, markers=True)
        st.plotly_chart(fig_overview, use_container_width=True)
    else:
        st.info("Add a 'Month' column and at least one numeric column to see trend charts here.")

# ===== TAB 2: INGREDIENTS =====
with tab_ingredients:
    st.subheader("Ingredient Insights")

    if ingredient_col in filtered_data.columns:
        # Top ingredients by first numeric column
        numeric_cols = filtered_data.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            y_col_ing = st.selectbox(
                "Numeric column for ingredient ranking",
                options=list(numeric_cols),
                index=0,
                key="ing_y_col",
            )

            ing_grouped = (
                filtered_data.groupby(ingredient_col, as_index=False)[y_col_ing]
                .sum()
                .sort_values(y_col_ing, ascending=False)
            )

            st.markdown(f"**Top 10 Ingredients by {y_col_ing}**")
            top10 = ing_grouped.head(10)

            c1, c2 = st.columns(2)
            with c1:
                fig_ing = px.bar(
                    top10,
                    x=ingredient_col,
                    y=y_col_ing,
                    title=f"Top 10 by {y_col_ing}",
                )
                st.plotly_chart(fig_ing, use_container_width=True)
            with c2:
                st.dataframe(top10, use_container_width=True)

            # If month + ingredient, show trend for selected ingredient
            if month_col:
                st.markdown("---")
                st.markdown("**Ingredient usage over time**")

                focus_ing = st.selectbox(
                    "Select ingredient for time-series view",
                    options=ing_grouped[ingredient_col].tolist(),
                )

                ing_ts = (
                    filtered_data[filtered_data[ingredient_col] == focus_ing]
                    .groupby(month_col, as_index=False)[y_col_ing]
                    .sum()
                    .sort_values(month_col)
                )

                fig_ing_ts = px.line(
                    ing_ts,
                    x=month_col,
                    y=y_col_ing,
                    title=f"{focus_ing} — {y_col_ing} over time",
                    markers=True,
                )
                st.plotly_chart(fig_ing_ts, use_container_width=True)
        else:
            st.warning("No numeric columns found to analyze ingredients.")
    else:
        st.warning("No ingredient-like column found. Update 'ingredient_col' to match your data.")

# ===== TAB 3: FORECAST (PLACEHOLDER) =====
with tab_forecast:
    st.subheader("Forecast & Reorder Suggestions")
    st.info(
        "This is where your predictive analysis friend will plug in their forecast "
        "(e.g., forecasted usage next month, suggested reorder quantities)."
    )
    st.write("For now, you can show a placeholder table or simple text here.")

# ===== TAB 4: SHIPPING (PLACEHOLDER) =====
with tab_shipping:
    st.subheader("Shipping & Supplier Performance")
    st.info(
        "This section will visualize shipment delays, supplier performance, and cost patterns "
        "using the shipping dataset."
    )
    st.write("Your shipping-focused teammate can provide a summary table/DataFrame to show here.")
