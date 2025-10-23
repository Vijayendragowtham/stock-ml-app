import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

st.set_page_config(page_title="Stock Value Prediction", layout="wide")
st.title("📊 Stock Value Prediction — Machine Learning App")

st.sidebar.header("1️⃣ Upload Your Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("### Dataset Preview")
    st.dataframe(df.head())
else:
    st.info("Using sample dataset: synthetic_stock_data.csv (upload your own to replace it)")
    df = pd.read_csv("synthetic_stock_data.csv")

st.write("Dataset Shape:", df.shape)

# ------------------------- PREPROCESSING ----------------------
st.header("2️⃣ Data Preprocessing")

# Handle missing values
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(df.mode().iloc[0])

# Encode categorical variables
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Feature scaling
scaler = StandardScaler()
num_cols = df.select_dtypes(include=np.number).columns.tolist()
df[num_cols] = scaler.fit_transform(df[num_cols])

st.success("✅ Data cleaned, encoded, and scaled successfully!")
st.header("3️⃣ Exploratory Data Analysis (EDA)")
if st.checkbox("Show Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    st.pyplot(fig)

if st.checkbox("Show Pairplot (first 4 numeric columns)"):
    sns.pairplot(df[num_cols[:4]])
    st.pyplot(plt.gcf())

st.header("4️⃣ Model Building & Evaluation")

target_col = st.selectbox("Select Target Column (Y)", df.columns)
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols]
y = df[target_col]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Evaluation Results")
st.write(f"**MAE:** {mae:.4f}")
st.write(f"**RMSE:** {rmse:.4f}")
st.write(f"**R²:** {r2:.4f}")

# Scatter plot
fig, ax = plt.subplots(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)


joblib.dump(model, "trained_model.joblib")

# ------------------------- RESULT & CONCLUSION ---------------
st.header("5️⃣ Result & Conclusion")

st.markdown(f"""
**Model:** Random Forest Regressor  
**Dataset:** {uploaded.name if uploaded else 'synthetic_stock_data.csv'}  
**Performance Summary:**
- MAE : {mae:.4f}
- RMSE: {rmse:.4f}
- R²  : {r2:.4f}
""")

if r2 > 0.8:
    st.success("Excellent model — explains most of the variance!")
elif r2 > 0.5:
    st.warning("Moderate performance — can be improved with feature engineering.")
else:
    st.error("Low performance — try tuning the model or using more features.")

st.download_button("📦 Download Trained Model", open("trained_model.joblib", "rb"), "trained_model.joblib")

st.markdown("---")
st.info("👩‍💻 Created & deployed using Streamlit. Run locally with: `streamlit run app.py`")

