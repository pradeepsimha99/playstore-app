# ==========================================================
# GOOGLE PLAY STORE ANALYTICS PROJECT
# PART 1 - FOUNDATION
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Google Play Store Analytics",
    page_icon="📱",
    layout="wide"
)


# ==========================================================
# THEME TOGGLE
# ==========================================================

# ==========================================================
# THEME TOGGLE
# ==========================================================

theme = st.sidebar.selectbox(
    "Theme",
    ["Light", "Dark"]
)

plotly_template = "plotly"

if theme == "Dark":

    plotly_template = "plotly_dark"

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #000;
            color: white;
        }

        .stButton > button,
        .stDownloadButton > button {
            background-color: #38BDF8;
            color: white;
            border: none;
            border-radius: 8px;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background-color: #0284C7;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

else:

    plotly_template = "plotly"


# ==========================================================
# CUSTOM HEADER
# ==========================================================

st.markdown(
    """
    <h1 style='text-align:center;'>
        📱 Google Play Store Analytics & Prediction System
    </h1>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# REQUIRED FILES CHECK
# ==========================================================

required_files = [
    "google_playstore_final_dataset.csv",
    "playstore_rating_model.pkl",
    "popular_app_model.pkl",
    "category_encoder.pkl"
]

missing_files = []

for file in required_files:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:

    st.error("❌ Missing Required Files")

    for file in missing_files:
        st.write(f"• {file}")

    st.stop()

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "google_playstore_final_dataset.csv"
    )

    return df


@st.cache_resource
def load_models():

    rating_model = joblib.load(
        "playstore_rating_model.pkl"
    )

    popular_model = joblib.load(
        "popular_app_model.pkl"
    )

    encoder = joblib.load(
        "category_encoder.pkl"
    )

    return rating_model, popular_model, encoder


try:

    df = load_data()

    numeric_cols = [
    "Rating",
    "Reviews",
    "Installs",
    "Price",
    "Size_KB",
    "Review_Count",
    "Positive_Percent",
    "Negative_Percent",
    "Neutral_Percent",
    "Avg_Sentiment_Polarity",
    "Avg_Sentiment_Subjectivity"
]

    for col in numeric_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    rating_model, popular_model, encoder = load_models()

except Exception as e:

    st.error(f"Loading Error: {e}")

    st.stop()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📱 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "📊 EDA Analysis",
        "⭐ Rating Predictor",
        "🔥 Popular App Predictor",
        "📱 App Explorer",
        "📈 Feature Importance"
    ]
)

# ==========================================================
# DASHBOARD PAGE
# ==========================================================

if page == "🏠 Dashboard":

    st.header("🏠 Dashboard")

    total_apps = len(df)

    avg_rating = round(
        df["Rating"].mean(),
        2
    )

    total_installs = int(
        df["Installs"].sum()
    )

    total_categories = df[
        "Category"
    ].nunique()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Apps",
            f"{total_apps:,}"
        )

    with col2:

        st.metric(
            "Average Rating",
            avg_rating
        )

    with col3:

        st.metric(
            "Total Installs",
            f"{total_installs:,}"
        )

    with col4:

        st.metric(
            "Categories",
            total_categories
        )

    st.divider()

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    st.download_button(
        label="⬇ Download Dataset",
        data=df.to_csv(index=False),
        file_name="google_playstore_final_dataset.csv",
        mime="text/csv"
    )

# ==========================================================
# EDA PAGE
# ==========================================================

elif page == "📊 EDA Analysis":

    st.header(
        "📊 Exploratory Data Analysis"
    )

    # --------------------------------------------------
    # TOP CATEGORIES
    # --------------------------------------------------

    st.subheader(
        "Top Categories"
    )

    category_counts = (
        df["Category"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    category_counts.columns = [
        "Category",
        "Count"
    ]

    fig1 = px.bar(
        category_counts,
        x="Category",
        y="Count",
        title="Top 10 Categories"
    )

    fig1.update_layout(
        template=plotly_template
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # --------------------------------------------------
    # TOP INSTALLS
    # --------------------------------------------------

    st.subheader(
        "Top Installed Categories"
    )

    install_counts = (
        df.groupby(
            "Category"
        )["Installs"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
        .reset_index()
    )

    fig2 = px.bar(
        install_counts,
        x="Category",
        y="Installs",
        title="Top Installed Categories"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ==========================================================
# PLACEHOLDERS
# ==========================================================

elif page == "⭐ Rating Predictor":

    st.header(
        "⭐ Rating Predictor"
    )

    st.write(
        "Predict the expected app rating using the trained Random Forest model."
    )

    categories = sorted(
        df["Category"].unique()
    )

    category = st.selectbox(
        "Category",
        categories,
        key="rating_category"
    )

    reviews = st.number_input(
        "Reviews",
        min_value=0.0,
        value=1000.0,
        key="rating_reviews"
    )

    installs = st.number_input(
        "Installs",
        min_value=0.0,
        value=100000.0,
        key="rating_installs"
    )

    price = st.number_input(
        "Price",
        min_value=0.0,
        value=0.0,
        key="rating_price"
    )

    size_kb = st.number_input(
        "Size (KB)",
        min_value=0.0,
        value=15000.0,
        key="rating_size"
    )

    polarity = st.slider(
        "Sentiment Polarity",
        -1.0,
        1.0,
        0.0,
        key="rating_polarity"
    )

    subjectivity = st.slider(
        "Sentiment Subjectivity",
        0.0,
        1.0,
        0.5,
        key="rating_subjectivity"
    )

    review_count = st.number_input(
        "Review Count",
        min_value=0.0,
        value=100.0,
        key="rating_review_count"
    )

    positive_percent = st.slider(
        "Positive %",
        0.0,
        100.0,
        50.0,
        key="rating_positive"
    )

    negative_percent = st.slider(
        "Negative %",
        0.0,
        100.0,
        10.0,
        key="rating_negative"
    )

    if st.button(
        "Predict Rating"
    ):

        try:

            encoded_category = encoder.transform(
                [category]
            )[0]

            features = np.array([
                [
                    encoded_category,
                    reviews,
                    installs,
                    price,
                    size_kb,
                    polarity,
                    subjectivity,
                    review_count,
                    positive_percent,
                    negative_percent
                ]
            ])

            prediction = rating_model.predict(
                features
            )[0]

            st.success(
                f"Predicted Rating: {prediction:.2f} ⭐"
            )

            report_df = pd.DataFrame({
                "Category": [category],
                "Predicted Rating": [round(prediction, 2)]
            })

            st.download_button(
                label="⬇ Download Rating Prediction",
                data=report_df.to_csv(index=False),
                file_name="rating_prediction.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )

elif page == "🔥 Popular App Predictor":

    st.header(
        "🔥 Popular App Predictor"
    )

    st.write(
        "Predict whether an app is likely to become popular."
    )

    categories = sorted(
        df["Category"].unique()
    )

    category = st.selectbox(
        "Category",
        categories,
        key="popular_category"
    )

    reviews = st.number_input(
        "Reviews",
        min_value=0.0,
        value=5000.0,
        key="popular_reviews"
    )

    price = st.number_input(
        "Price",
        min_value=0.0,
        value=0.0,
        key="popular_price"
    )

    size_kb = st.number_input(
        "Size (KB)",
        min_value=0.0,
        value=15000.0,
        key="popular_size"
    )

    polarity = st.slider(
        "Sentiment Polarity",
        -1.0,
        1.0,
        0.0,
        key="popular_polarity"
    )

    subjectivity = st.slider(
        "Sentiment Subjectivity",
        0.0,
        1.0,
        0.5,
        key="popular_subjectivity"
    )

    review_count = st.number_input(
        "Review Count",
        min_value=0.0,
        value=100.0,
        key="popular_review_count"
    )

    positive_percent = st.slider(
        "Positive %",
        0.0,
        100.0,
        50.0,
        key="popular_positive"
    )

    negative_percent = st.slider(
        "Negative %",
        0.0,
        100.0,
        10.0,
        key="popular_negative"
    )

    if st.button(
        "Predict Popularity"
    ):

        try:

            encoded_category = encoder.transform(
                [category]
            )[0]

            features = np.array([
                [
                    encoded_category,
                    reviews,
                    price,
                    size_kb,
                    polarity,
                    subjectivity,
                    review_count,
                    positive_percent,
                    negative_percent
                ]
            ])

            prediction = popular_model.predict(
                features
            )[0]

            probability = (
                popular_model
                .predict_proba(features)[0][1]
            )

            confidence = round(
                probability * 100,
                2
            )

            import plotly.graph_objects as go

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    title={
                        "text":
                        "Popularity Probability"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        }
                    }
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            if prediction == 1:

                st.success(
                    f"🔥 Popular App ({confidence}% confidence)"
                )

            else:

                st.warning(
                    f"⚠️ Not Popular App ({100-confidence:.2f}% confidence)"
                )

            report_df = pd.DataFrame({
                "Category": [category],
                "Probability": [confidence],
                "Prediction": [
                    "Popular"
                    if prediction == 1
                    else "Not Popular"
                ]
            })

            st.download_button(
                label="⬇ Download Popularity Report",
                data=report_df.to_csv(index=False),
                file_name="popularity_prediction.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )

elif page == "📱 App Explorer":

    st.header(
        "📱 App Explorer"
    )

    st.write(
        "Search any app from the dataset."
    )

    search_app = st.text_input(
        "🔍 Search App Name"
    )

    if not search_app:

        st.info(
            "Start typing an app name..."
        )

    else:

        filtered_apps = df[
            df["App"]
            .str.contains(
                search_app,
                case=False,
                na=False
            )
        ]["App"].unique()

        st.write(
            f"Found {len(filtered_apps)} matching apps"
        )

        if len(filtered_apps) == 0:

            st.error(
                "No matching apps found."
            )

        else:

            selected_app = st.selectbox(
                "Choose App",
                sorted(filtered_apps)
            )

            app_data = df[
                df["App"] == selected_app
            ]

            if not app_data.empty:

                row = app_data.iloc[0]

                # ======================================
                # TOP METRICS
                # ======================================

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Rating",
                        round(
                            row["Rating"],
                            2
                        )
                    )

                with col2:

                    st.metric(
                        "Installs",
                        f"{int(row['Installs']):,}"
                    )

                with col3:

                    st.metric(
                        "Price",
                        f"${row['Price']}"
                    )

                st.divider()

                # ======================================
                # APP DETAILS
                # ======================================

                st.subheader(
                    "Application Details"
                )

                st.write(
                    f"**App Name:** {row['App']}"
                )

                st.write(
                    f"**Category:** {row['Category']}"
                )

                st.write(
                    f"**Review Count:** {int(row['Review_Count'])}"
                )

                # ======================================
                # SENTIMENT METRICS
                # ======================================

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Positive %",
                        f"{row['Positive_Percent']:.2f}%"
                    )

                with col2:

                    st.metric(
                        "Negative %",
                        f"{row['Negative_Percent']:.2f}%"
                    )

                with col3:

                    st.metric(
                        "Neutral %",
                        f"{row['Neutral_Percent']:.2f}%"
                    )

                st.write(
                    f"**Sentiment Polarity:** {row['Avg_Sentiment_Polarity']:.3f}"
                )

                st.write(
                    f"**Sentiment Subjectivity:** {row['Avg_Sentiment_Subjectivity']:.3f}"
                )

                st.divider()

                # ======================================
                # DEBUG VALUES
                # ======================================

                st.subheader(
                    "Sentiment Data"
                )

                sentiment_df = pd.DataFrame({

                    "Sentiment": [
                        "Positive",
                        "Negative",
                        "Neutral"
                    ],

                    "Count": [
                        row["Positive"],
                        row["Negative"],
                        row["Neutral"]
                    ]

                })

                sentiment_df["Count"] = pd.to_numeric(
                    sentiment_df["Count"],
                    errors="coerce"
                )

                st.dataframe(
                    sentiment_df,
                    use_container_width=True
                )

                # ======================================
                # DOWNLOAD APP DATA
                # ======================================

                st.download_button(
                    label="⬇ Download App Data",
                    data=app_data.to_csv(index=False),
                    file_name=f"{row['App']}.csv",
                    mime="text/csv",
                )

elif page == "📈 Feature Importance":

    st.header(
        "📈 Feature Importance Analysis"
    )

    st.write(
        "Features affecting Rating Prediction"
    )

    feature_data = pd.DataFrame({

        "Feature": [
            "Reviews",
            "Size_KB",
            "Category",
            "Installs",
            "Price",
            "Avg_Sentiment_Polarity",
            "Positive_Percent",
            "Review_Count",
            "Avg_Sentiment_Subjectivity",
            "Negative_Percent"
        ],

        "Importance": [
            0.353102,
            0.311050,
            0.169438,
            0.113847,
            0.032773,
            0.005670,
            0.003886,
            0.003590,
            0.003346,
            0.003298
        ]
    })

    feature_data = feature_data.sort_values(
        "Importance",
        ascending=True
    )

    fig = px.bar(
        feature_data,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        feature_data,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "Business Insights"
    )

    st.success(
        """
        Reviews are the strongest predictor of app ratings.

        App Size is the second most important factor.

        Category and Installs also influence ratings.

        Sentiment features contribute less than expected.
        """
    )

if page == "🏠 Dashboard":

    st.divider()

    st.subheader(
        "Quick Dataset Insights"
    )

    st.info(
        f"""
        Total Apps: {len(df):,}

        Average Rating: {df['Rating'].mean():.2f}

        Highest Rating: {df['Rating'].max()}

        Categories: {df['Category'].nunique()}
        """
    )