import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
model_churn_label = joblib.load("streamlitapp/churn_label_model.pkl")
model_churn = joblib.load("streamlitapp/churn_model.pkl")
model_churn_rf = joblib.load("streamlitapp/churn_rf_model.pkl")

# Load scalers
scaler_completion = joblib.load("streamlitapp/scaler_completion.pkl")
scaler_unique = joblib.load("streamlitapp/scaler_unique.pkl")
scaler_completion_rf = joblib.load("streamlitapp/scaler_completion_rf.pkl")
scaler_unique_rf = joblib.load("streamlitapp/scaler_unique_rf.pkl")

# Preprocessing functions
def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['completion_scaled'] = scaler_completion.transform(df[['avg_completion_rate']])
    df['log_unique_videos'] = np.log1p(df['unique_videos_watched'])
    df['unique_scaled'] = scaler_unique.transform(df[['log_unique_videos']])
    return df[['completion_scaled', 'unique_scaled']]

def preprocess_features_robust(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['completion_scaled'] = scaler_completion_rf.transform(df[['avg_completion_rate']])
    df['unique_scaled'] = scaler_unique_rf.transform(df[['unique_videos_watched']])
    return df[['completion_scaled', 'unique_scaled']]

# Streamlit UI
st.title("📊 Learner Churn Prediction App")

model_type = st.sidebar.selectbox("Select Prediction Model", ["churn", "churn_label", "churn_rf"])
mode = st.radio("Choose prediction mode:", ["Single Learner", "Batch Prediction (CSV Upload)"])

if mode == "Single Learner":
    st.subheader("🔍 Predict Single Learner")
    avg_completion_rate = st.number_input("Average Completion Rate (%)", min_value=0.0, max_value=100.0, value=45.0)
    unique_videos_watched = st.number_input("Number of Unique Videos Watched", min_value=0, value=5)

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            'avg_completion_rate': avg_completion_rate,
            'unique_videos_watched': unique_videos_watched
        }])

        if model_type == "churn_label":
            features = preprocess_features(input_df)
            prediction = model_churn_label.predict(features)[0]

        elif model_type == "churn":
            features = preprocess_features(input_df)
            prediction = model_churn.predict(features)[0]

        elif model_type == "churn_rf":
            features = preprocess_features_robust(input_df)
            prediction = model_churn_rf.predict(features)[0]

        st.success(f"✅ Prediction: {'Churn' if prediction == 1 else 'Not Churn'}")

else:
    st.subheader("📁 Batch Prediction with CSV Upload")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if st.button("Predict Batch"):
            if model_type == "churn_label":
                features = preprocess_features(df)
                df['prediction'] = model_churn_label.predict(features)
            elif model_type == "churn":
                features = preprocess_features(df)
                df['prediction'] = model_churn.predict(features)
            elif model_type == "churn_rf":
                features = preprocess_features_robust(df)
                df['prediction'] = model_churn_rf.predict(features)

            st.success("✅ Batch prediction completed.")
            st.dataframe(df)
