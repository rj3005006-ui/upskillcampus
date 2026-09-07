import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Ruchi3056/traffic-prediction-model",
    filename="traffic_prediction_model.pkl"
)

model = joblib.load(model_path)
features = joblib.load("model_features.pkl")

df = pd.read_csv("train_aWnotuB.csv")
feature_df = pd.read_csv("feature_importance.csv")

df["DateTime"] = pd.to_datetime(df["DateTime"])
df["Hour"] = df["DateTime"].dt.hour
traffic_hour = df.groupby("Hour")["Vehicles"].mean().reset_index()


st.set_page_config(
    page_title="Traffic Flow Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background-color: #FFFFFF;
    color :#FFFFFF

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #1E293B;
    color : #FFFFFF
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1E3A8A;
    color : #FFFFFFF
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: black;
}

</style>
""", unsafe_allow_html=True)

st.title("🚗 Smart Traffic Flow Prediction Dashboard")
st.caption("AI-Powered Traffic Analytics using Machine Learning")

st.markdown("""
### 🚀 Welcome!

This dashboard uses a **Random Forest Regression** model to predict traffic volume based on:

- 🚦 Junction
- 📅 Date
- ⏰ Time
- 📆 Day of Week

Use the sidebar to make a prediction and explore historical traffic trends below.
""")


# ==========================
# Sidebar
# ==========================

st.sidebar.header("⚙️ Prediction Inputs")

junction = st.sidebar.selectbox(
    "Select Junction",
    [1, 2, 3, 4]
)

year = st.sidebar.selectbox(
    "Select Year",
    [2015, 2016, 2017]
)

month = st.sidebar.selectbox(
    "Select Month",
    list(range(1, 13))
)

day = st.sidebar.slider(
    "Select Day",
    1,
    31,
    15
)

hour = st.sidebar.slider(
    "Select Hour",
    0,
    23,
    12
)

weekday = st.sidebar.selectbox(
    "Select Day of Week",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

predict_button = st.sidebar.button(
    "🔮 Predict Traffic",
    use_container_width=True
)

predicted_vehicles = "--"
traffic_status = "--"
traffic_emoji = "🚦"

if predict_button:

    input_data = {
        "Junction": junction,
        "Year": year,
        "Month": month,
        "Day": day,
        "Hour": hour,

        "DayOfWeek_Monday": 0,
        "DayOfWeek_Saturday": 0,
        "DayOfWeek_Sunday": 0,
        "DayOfWeek_Thursday": 0,
        "DayOfWeek_Tuesday": 0,
        "DayOfWeek_Wednesday": 0
    }

    if weekday != "Friday":
      input_data[f"DayOfWeek_{weekday}"] = 1

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)

    predicted_vehicles = round(prediction[0])

    if predicted_vehicles < 20:
        traffic_status = "Low"
        traffic_emoji = "🟢"

    elif predicted_vehicles < 50:
        traffic_status = "Moderate"
        traffic_emoji = "🟡"

    else:
        traffic_status = "Heavy"
        traffic_emoji = "🔴"



st.divider()

col1, col2 = st.columns(2)

with col1:

    st.subheader("📈 Average Traffic by Hour")

    traffic_hour = (
        df.groupby("Hour")["Vehicles"]
          .mean()
          .reset_index()
    )

    fig1 = px.line(
        traffic_hour,
        x="Hour",
        y="Vehicles",
        markers=True,
        title=""
    )

    st.plotly_chart(fig1, use_container_width=True)
    
with col2:

    st.subheader("🚦 Average Traffic by Junction")

    traffic_junction = (
        df.groupby("Junction")["Vehicles"]
          .mean()
          .reset_index()
    )

    fig2 = px.bar(
        traffic_junction,
        x="Junction",
        y="Vehicles",
        title="",
        text_auto=".1f"
    )

    st.plotly_chart(fig2, use_container_width=True)
    
st.divider()

col3, col4 = st.columns(2)
with col3:

    st.subheader("📅 Average Traffic by Weekday")

    # Create weekday names
    df["DayOfWeek"] = df["DateTime"].dt.day_name()

    # Maintain correct weekday order
    weekday_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    traffic_day = (
        df.groupby("DayOfWeek")["Vehicles"]
        .mean()
        .reindex(weekday_order)
        .reset_index()
    )

    fig3 = px.line(
        traffic_day,
        x="DayOfWeek",
        y="Vehicles",
        markers=True,
        title="Average Traffic by Weekday"
    )

    fig3.update_layout(
        xaxis_title="Day",
        yaxis_title="Average Vehicles",
        hovermode="x unified"
    )

    st.plotly_chart(fig3, use_container_width=True)
    
with col4:

    st.subheader("📆 Average Traffic by Month")

    df["Month"] = df["DateTime"].dt.month

    traffic_month = (
        df.groupby("Month")["Vehicles"]
        .mean()
        .reset_index()
    )

    fig4 = px.line(
        traffic_month,
        x="Month",
        y="Vehicles",
        markers=True,
        title="Average Traffic by Month"
    )

    fig4.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Vehicles",
        hovermode="x unified"
    )

    st.plotly_chart(fig4, use_container_width=True)

st.divider()

st.subheader("🌳 Feature Importance")

fig5 = px.bar(
    feature_df,
    x="Importance",
    y="Feature",
    orientation="h",
    text_auto=".3f",
    title="Features Influencing Traffic Prediction"
)

fig5.update_layout(
    yaxis=dict(categoryorder="total ascending")
)

st.plotly_chart(fig5, use_container_width=True)


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🚗 Predicted Vehicles",
        predicted_vehicles
    )

with col2:
    st.metric(
        "📈 Model Accuracy",
        "71.2%"
    )

with col3:
    st.metric(
        "🚦 Traffic Status",
        f"{traffic_emoji} {traffic_status}"
    )


st.divider()

st.subheader("📋 Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", df.shape[1])
col3.metric("Junctions", df["Junction"].nunique())
col4.metric("Years", df["DateTime"].dt.year.nunique())

st.divider()

st.subheader("🤖 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", "10.83")
col2.metric("R² Score", "0.712")
col3.metric("Algorithm", "Random Forest")

st.divider()

st.subheader("📖 About This Project")

st.write("""
This dashboard predicts traffic volume using a **Random Forest Regression** model
trained on historical traffic data.

### Technologies Used
- Python
- Pandas
- Scikit-learn
- Plotly
- Streamlit

### Input Features
- Junction
- Year
- Month
- Day
- Hour
- Day of Week

### Output
- Predicted number of vehicles
- Traffic status (Low, Moderate, Heavy)
""")


st.divider()