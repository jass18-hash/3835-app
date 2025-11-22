import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)

# --------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Birds Activity Model – XAI Dashboard",
    layout="wide"
)

# --------- SIDEBAR ----------
st.sidebar.title("🦉 Birds XAI Dashboard")
st.sidebar.write("Use this panel to explore the model and findings.")

show_data = st.sidebar.checkbox("Show Raw Data", value=False)

# --------- LOAD DATA ----------
@st.cache_data
def load_data():
    return pd.read_csv("Birds_cleaned.csv")

df = load_data()

# --------- MAIN TITLE ----------
st.title("Birds Activity Model – XAI Dashboard")
st.write("This dashboard trains a Logistic Regression model and visualizes key XAI elements.")

if show_data:
    st.subheader("Dataset Preview")
    st.write(df.head())

# --------- PREPROCESSING ----------
df["y"] = df["active__recovered"].map({"Y": 1, "N": 0}).fillna(0).astype(int)

# Handle rare categories
model_counts = df["active__model"].value_counts()
rare_models = model_counts[model_counts < 5].index
df["active__model"] = df["active__model"].replace(rare_models, "Other")

# One-hot encode
df = pd.get_dummies(df, columns=["active__model"], prefix="model", dtype=int)

# Feature selection
X = df[[c for c in df.columns if c.startswith("model_")]]
y = df["y"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------- MODEL TRAINING ----------
param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")

grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_log_reg = grid_search.best_estimator_
y_pred_best = best_log_reg.predict(X_test)

# --------- METRICS ----------
accuracy = accuracy_score(y_test, y_pred_best)
precision = precision_score(y_test, y_pred_best)
recall = recall_score(y_test, y_pred_best)
f1 = f1_score(y_test, y_pred_best)

# --------- FEATURE IMPORTANCE ----------
feature_names = X_train.columns
coefficients = best_log_reg.coef_.ravel()

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute Importance": np.abs(coefficients)
}).sort_values("Absolute Importance", ascending=False)

recovery_features = ["model_recovery_rate_overall", "model_month_recovery_rate"]
recovery_df = coef_df[coef_df["Feature"].isin(recovery_features)]
other_df = coef_df[~coef_df["Feature"].isin(recovery_features)]

# --------- TABS ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Model Performance",
    "🔥 Feature Importance",
    "📌 Recovery Features",
    "📉 Other Models",
    "📘 EDA Findings"
])

# --------- TAB 1: MODEL PERFORMANCE ----------
with tab1:
    st.header("📊 Model Performance (Optimized Logistic Regression)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")
    col4.metric("F1-score", f"{f1:.3f}")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 3))
    ConfusionMatrixDisplay.from_estimator(best_log_reg, X_test, y_test, cmap="Blues", ax=ax)
    st.pyplot(fig)

# --------- TAB 2: COEFFICIENT IMPORTANCE ----------
with tab2:
    st.header("🔥 Coefficient-Based Feature Importance")
    st.dataframe(coef_df)

    st.subheader("Top 10 Most Important Features")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(coef_df["Feature"].head(10)[::-1], coef_df["Absolute Importance"].head(10)[::-1])
    ax.set_xlabel("Absolute Coefficient Value")
    st.pyplot(fig)

# --------- TAB 3: RECOVERY FEATURES ----------
with tab3:
    st.header("📌 Recovery Feature Comparison")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(recovery_df["Feature"], recovery_df["Coefficient"],
           color=["skyblue", "orange"], edgecolor="black")
    ax.set_ylabel("Coefficient Value")
    st.pyplot(fig)

# --------- TAB 4: OTHER FEATURES ----------
with tab4:
    st.header("📉 Other Model Features Coefficients")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(other_df["Feature"], other_df["Coefficient"], marker='o')
    plt.xticks(rotation=90)
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

# --------- TAB 5: EDA FINDINGS (IMAGES) ----------
with tab5:
    st.header("📘 EDA Findings Summary")
    st.write("These are the key EDA observations based on the dataset.")

    st.write("### 🔹 Image 1")
    st.image("image1.png")

    st.write("### 🔹 Image 2")
    st.image("image2.png")

    st.write("### 🔹 Image 3")
    st.image("image3.png")

    st.write("### 🔹 Image 4")
    st.image("image4.png")

    st.write("### 🔹 Image 5")
    st.image("image5.png")

st.success("Dashboard loaded successfully!")
