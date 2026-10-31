import io
import json
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, r2_score, roc_auc_score
)
from sklearn.model_selection import (
    RandomizedSearchCV, cross_val_score, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AutoML Express", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family: Inter, sans-serif;}
.block-container {padding-top: 1.8rem; padding-bottom: 3rem;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px;
background: linear-gradient(120deg,#101c3a,#214b75,#367b86); color:white;
margin-bottom:1.2rem; box-shadow:0 10px 30px rgba(20,50,90,.18);}
.hero h1 {font-size:2.2rem; margin:0 0 .35rem 0; color:white;}
.hero p {margin:0; color:#e5f1ff;}
div.stButton > button[kind="primary"] {border-radius:10px; font-weight:700; min-height:2.8rem;}
div[data-testid="stMetric"] {background:rgba(120,150,180,.08); padding:12px; border-radius:12px;}
div[data-testid="stExpander"] {border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🤖 AutoML Express</h1>
  <p>Explore • Clean • Select Features • Train • Evaluate • Download your report</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------- Utility functions -----------------------------
def load_dataset(file):
    filename = file.name.lower()
    if filename.endswith(".csv"):
        try:
            file.seek(0)
            return pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, encoding="latin-1")
    file.seek(0)
    return pd.read_excel(file)

def unique_column_names(columns):
    counts, output = {}, []
    for col in columns:
        name = str(col).strip() or "unnamed"
        counts[name] = counts.get(name, 0) + 1
        output.append(name if counts[name] == 1 else f"{name}_{counts[name]}")
    return output

def get_models(problem):
    if problem == "Regression":
        return {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
            "Extra Trees": ExtraTreesRegressor(n_estimators=150, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Support Vector Regression": SVR(),
        }
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(probability=True),
    }

def build_preprocessor(X, strategies):
    transformers = []
    for col in X.columns:
        strategy = strategies.get(col, "median" if pd.api.types.is_numeric_dtype(X[col]) else "most_frequent")
        if strategy == "drop rows":
            strategy = "median" if pd.api.types.is_numeric_dtype(X[col]) else "most_frequent"
        if strategy == "constant":
            fill_value = "Missing" if not pd.api.types.is_numeric_dtype(X[col]) else 0
            imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
        else:
            imputer = SimpleImputer(strategy=strategy)
        steps = [("imputer", imputer)]
        if pd.api.types.is_numeric_dtype(X[col]):
            steps.append(("scaler", StandardScaler()))
        else:
            steps.append(("encoder", OneHotEncoder(handle_unknown="ignore")))
        transformers.append((f"col_{len(transformers)}", Pipeline(steps), [col]))
    if not transformers:
        raise ValueError("No usable feature columns remain.")
    return ColumnTransformer(transformers=transformers, remainder="drop")

def metric_values(problem, y_true, pred, model=None, X_test=None):
    if problem == "Regression":
        mse = mean_squared_error(y_true, pred)
        return {
            "R²": round(float(r2_score(y_true, pred)), 4),
            "MAE": round(float(mean_absolute_error(y_true, pred)), 4),
            "MSE": round(float(mse), 4),
            "RMSE": round(float(mse ** 0.5), 4),
        }
    values = {
        "Accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "Precision (weighted)": round(float(precision_score(y_true, pred, average="weighted", zero_division=0)), 4),
        "Recall (weighted)": round(float(recall_score(y_true, pred, average="weighted", zero_division=0)), 4),
        "F1 (weighted)": round(float(f1_score(y_true, pred, average="weighted", zero_division=0)), 4),
    }
    if model is not None and X_test is not None and hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_test)
            classes = getattr(model, "classes_", [])
            if len(classes) == 2:
                values["ROC-AUC"] = round(float(roc_auc_score(y_true, probabilities[:, 1])), 4)
            elif len(classes) > 2:
                values["ROC-AUC"] = round(float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted", labels=classes)), 4)
        except Exception:
            pass
    return values

def tuning_space(name, problem):
    if problem == "Regression":
        spaces = {
            "Ridge Regression": {"model__alpha": [0.01, 0.1, 1, 10, 100]},
            "Decision Tree": {"model__max_depth": [None, 3, 5, 10, 20], "model__min_samples_split": [2, 5, 10]},
            "Random Forest": {"model__n_estimators": [100, 150, 250], "model__max_depth": [None, 5, 10, 20]},
            "Extra Trees": {"model__n_estimators": [100, 150, 250], "model__max_depth": [None, 5, 10, 20]},
            "Gradient Boosting": {"model__n_estimators": [50, 100, 150], "model__learning_rate": [0.01, 0.05, 0.1]},
            "K-Nearest Neighbors": {"model__n_neighbors": [3, 5, 7, 9], "model__weights": ["uniform", "distance"]},
            "Support Vector Regression": {"model__C": [0.1, 1, 10, 100], "model__kernel": ["linear", "rbf"]},
        }
    else:
        spaces = {
            "Logistic Regression": {"model__C": [0.1, 1, 10, 100]},
            "Decision Tree": {"model__max_depth": [None, 3, 5, 10, 20], "model__min_samples_split": [2, 5, 10]},
            "Random Forest": {"model__n_estimators": [100, 150, 250], "model__max_depth": [None, 5, 10, 20]},
            "Extra Trees": {"model__n_estimators": [100, 150, 250], "model__max_depth": [None, 5, 10, 20]},
            "Gradient Boosting": {"model__n_estimators": [50, 100, 150], "model__learning_rate": [0.01, 0.05, 0.1]},
            "K-Nearest Neighbors": {"model__n_neighbors": [3, 5, 7, 9], "model__weights": ["uniform", "distance"]},
            "Support Vector Classifier": {"model__C": [0.1, 1, 10, 100], "model__kernel": ["linear", "rbf"]},
        }
    return spaces.get(name, {})

def pdf_report(report_data, results_df, cleaning_summary):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=38, leftMargin=38, topMargin=38, bottomMargin=38)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("AutoML Express — Model Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph(f"Generated: {report_data['generated_at']}", styles["Normal"]),
        Paragraph(f"Dataset: {report_data['dataset_name']}", styles["Normal"]),
        Paragraph(f"Prediction target: {report_data['target']}", styles["Normal"]),
        Paragraph(f"Problem type: {report_data['problem_type']}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Dataset and cleaning summary", styles["Heading2"]),
    ]
    clean_rows = [["Item", "Details"]] + [[str(k), str(v)] for k, v in cleaning_summary.items()]
    clean_table = Table(clean_rows, colWidths=[175, 330], repeatRows=1)
    clean_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#24476b")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .4, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef3f8")]),
    ]))
    story += [clean_table, Spacer(1, 14), Paragraph("Model comparison", styles["Heading2"])]
    table_data = [list(results_df.columns)] + results_df.astype(str).values.tolist()
    col_width = 505 / max(1, len(results_df.columns))
    result_table = Table(table_data, colWidths=[col_width] * len(results_df.columns), repeatRows=1)
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#24476b")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .35, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef3f8")]),
    ]))
    story += [result_table, Spacer(1, 14), Paragraph("Final selected model", styles["Heading2"])]
    story.append(Paragraph(f"Model: {report_data['best_model']}", styles["Normal"]))
    for key, value in report_data["final_metrics"].items():
        story.append(Paragraph(f"{key}: {value}", styles["Normal"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Note: Metrics are based on the held-out test split. Validate the model further before production use.", styles["Italic"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ----------------------------- Upload -----------------------------
uploaded = st.file_uploader("📂 Upload dataset (CSV or XLSX)", type=["csv", "xlsx"])
if uploaded is None:
    st.info("Start by uploading a dataset. Your data is processed during this app session.")
    st.stop()

try:
    raw_df = load_dataset(uploaded)
except Exception as exc:
    st.error(f"Could not read file: {exc}")
    st.stop()

raw_df.columns = unique_column_names(raw_df.columns)
raw_df = raw_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
if raw_df.empty or raw_df.shape[1] < 2:
    st.error("Dataset must contain rows and at least two columns.")
    st.stop()

st.success(f"Dataset loaded: {uploaded.name}")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows", f"{len(raw_df):,}")
k2.metric("Columns", raw_df.shape[1])
k3.metric("Duplicate rows", f"{int(raw_df.duplicated().sum()):,}")
k4.metric("Missing cells", f"{int(raw_df.isna().sum().sum()):,}")

tab_eda, tab_clean, tab_model, tab_results = st.tabs(["📊 EDA", "🧹 Data Cleaning", "🧠 Model Setup", "🏆 Results & Reports"])

# ----------------------------- EDA -----------------------------
with tab_eda:
    st.subheader("Dataset preview")
    st.dataframe(raw_df.head(20), use_container_width=True)
    info = pd.DataFrame({
        "Column": raw_df.columns,
        "Data type": raw_df.dtypes.astype(str).values,
        "Missing": raw_df.isna().sum().values,
        "Missing %": (raw_df.isna().mean().values * 100).round(2),
        "Unique values": raw_df.nunique(dropna=True).values,
    })
    st.subheader("Column profile")
    st.dataframe(info, use_container_width=True, hide_index=True)
    with st.expander("Statistical summary"):
        st.dataframe(raw_df.describe(include="all").T, use_container_width=True)
    num_cols = raw_df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        col = st.selectbox("Choose a numeric column to view its distribution", num_cols)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        sns.histplot(raw_df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
        plt.close(fig)
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.heatmap(raw_df[num_cols].corr(), cmap="coolwarm", annot=len(num_cols) <= 10, fmt=".2f", ax=ax)
            ax.set_title("Correlation heatmap")
            st.pyplot(fig)
            plt.close(fig)
    missing_counts = raw_df.isna().sum()
    if (missing_counts > 0).any():
        fig, ax = plt.subplots(figsize=(9, 3.5))
        missing_counts[missing_counts > 0].sort_values(ascending=False).plot(kind="bar", ax=ax)
        ax.set_title("Missing values by column")
        ax.set_ylabel("Count")
        plt.xticks(rotation=40, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ----------------------------- Cleaning controls -----------------------------
with tab_clean:
    st.subheader("Choose how to handle duplicate rows")
    duplicate_count = int(raw_df.duplicated().sum())
    st.write(f"Detected duplicate rows: **{duplicate_count}**")
    duplicate_action = st.radio(
        "Duplicate-row action",
        ["Keep duplicates", "Remove duplicate rows"],
        horizontal=True,
        disabled=(duplicate_count == 0),
    )
    st.subheader("Missing-value handling — choose a method for each column")
    missing_methods = {}
    missing_cols = raw_df.columns[raw_df.isna().any()].tolist()
    if not missing_cols:
        st.success("No missing values detected.")
    else:
        st.caption("Mean/median are for numeric columns. Mode is suitable for numeric or categorical columns.")
        for col in missing_cols:
            numeric = pd.api.types.is_numeric_dtype(raw_df[col])
            choices = ["Drop rows", "Drop column", "Mode", "Constant"]
            if numeric:
                choices = ["Drop rows", "Drop column", "Mean", "Median", "Mode", "Constant"]
            default = "Median" if numeric else "Mode"
            choice = st.selectbox(
                f"{col} — {int(raw_df[col].isna().sum())} missing ({raw_df[col].isna().mean()*100:.1f}%)",
                choices, index=choices.index(default), key=f"missing_{col}"
            )
            missing_methods[col] = choice.lower()
    st.subheader("Optional automatic column removal")
    threshold = st.slider("Drop feature columns when missing percentage is at least (%)", 0, 100, 100, 5)
    st.caption("100% means only completely empty columns are removed automatically. Choose a lower threshold to remove sparse columns.")
    st.subheader("Cleaned data preview")
    cleaned_preview = raw_df.copy()
    if duplicate_action == "Remove duplicate rows":
        cleaned_preview = cleaned_preview.drop_duplicates()
    drop_by_missing = [c for c, method in missing_methods.items() if method == "drop column"]
    threshold_drop = [c for c in cleaned_preview.columns if cleaned_preview[c].isna().mean() * 100 >= threshold and cleaned_preview[c].isna().any()] if threshold < 100 else []
    drop_cols_preview = list(dict.fromkeys(drop_by_missing + threshold_drop))
    cleaned_preview = cleaned_preview.drop(columns=drop_cols_preview, errors="ignore")
    drop_row_cols = [c for c, method in missing_methods.items() if method == "drop rows" and c in cleaned_preview.columns]
    if drop_row_cols:
        cleaned_preview = cleaned_preview.dropna(subset=drop_row_cols)
    st.write(f"Preview after duplicate and drop choices: **{len(cleaned_preview):,} rows × {len(cleaned_preview.columns)} columns**")
    st.dataframe(cleaned_preview.head(15), use_container_width=True)
    st.info("Imputation is applied inside the ML pipeline after train/test splitting, helping prevent data leakage. The preview above reflects row/column removals; imputed values are handled during training.")

# ----------------------------- Model setup -----------------------------
with tab_model:
    st.subheader("Choose target and features")
    target = st.selectbox("🎯 Target column (what should the model predict?)", raw_df.columns.tolist())
    all_feature_candidates = [c for c in raw_df.columns if c != target]
    selected_features = st.multiselect(
        "Select features to use (unselected columns will be excluded)",
        all_feature_candidates,
        default=all_feature_candidates,
    )
    st.write(f"Selected features: **{len(selected_features)} / {len(all_feature_candidates)}**")
    problem_guess = "Classification"
    if pd.api.types.is_numeric_dtype(raw_df[target]) and raw_df[target].nunique(dropna=True) > 10:
        problem_guess = "Regression"
    problem = st.radio("Problem type", ["Classification", "Regression"], index=0 if problem_guess == "Classification" else 1, horizontal=True)
    st.subheader("Training options")
    split = st.slider("Test data (%)", 10, 40, 20, 5)
    use_cv = st.checkbox("Perform cross-validation", value=True)
    cv_folds = st.selectbox("Cross-validation folds", [3, 5, 10], index=0, disabled=not use_cv)
    models = get_models(problem)
    selected_model_names = st.multiselect("Choose algorithms (leave all selected to compare all)", list(models.keys()), default=list(models.keys()))
    st.subheader("Evaluation metrics")
    if problem == "Regression":
        metric_choices = ["R²", "MAE", "MSE", "RMSE"]
        default_metrics = metric_choices
        sort_metric = st.selectbox("Metric used to rank models", ["RMSE", "MAE", "MSE", "R²"], index=0)
    else:
        metric_choices = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)", "ROC-AUC"]
        default_metrics = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)"]
        sort_metric = st.selectbox("Metric used to rank models", ["F1 (weighted)", "Accuracy", "Precision (weighted)", "Recall (weighted)", "ROC-AUC"])
    chosen_metrics = st.multiselect("Metrics to include in evaluation/report", metric_choices, default=default_metrics)
    show_confusion = st.checkbox("Show confusion matrix (classification only)", value=True, disabled=(problem != "Classification"))
    show_class_report = st.checkbox("Show detailed classification report", value=False, disabled=(problem != "Classification"))
    st.subheader("Optional tuning")
    tune_best = st.checkbox("Tune the best model after comparison", value=False)
    tune_iterations = st.slider("Randomized-search iterations", 5, 25, 10, disabled=not tune_best)
    train_clicked = st.button("🚀 Train selected models", type="primary", use_container_width=True)

    if train_clicked:
        if not selected_features:
            st.error("Please select at least one feature.")
        elif not selected_model_names:
            st.error("Please select at least one algorithm.")
        elif not chosen_metrics:
            st.error("Please select at least one evaluation metric.")
        else:
            work_df = raw_df.copy()
            if duplicate_action == "Remove duplicate rows":
                work_df = work_df.drop_duplicates()
            work_df = work_df.drop(columns=[c for c in drop_cols_preview if c != target], errors="ignore")
            usable_features = [c for c in selected_features if c in work_df.columns]
            if not usable_features:
                st.error("No selected features remain after cleaning choices.")
            else:
                target_missing_before = int(work_df[target].isna().sum())
                work_df = work_df.dropna(subset=[target])
                X = work_df[usable_features].copy()
                y = work_df[target].copy()
                # Convert datetime columns into numeric timestamps for ML.
                for c in X.columns:
                    if pd.api.types.is_datetime64_any_dtype(X[c]):
                        X[c] = X[c].map(lambda v: v.timestamp() if pd.notna(v) else float("nan"))
                # Treat infinite numeric values as missing.
                X = X.replace([float("inf"), float("-inf")], float("nan"))
                # Remove rows selected for drop-row treatment.
                row_drop_cols = [c for c, method in missing_methods.items() if method == "drop rows" and c in X.columns]
                if row_drop_cols:
                    keep = X[row_drop_cols].notna().all(axis=1)
                    X, y = X.loc[keep], y.loc[keep]
                # Remove any columns that are fully missing after row filtering.
                all_missing = X.columns[X.isna().all()].tolist()
                X = X.drop(columns=all_missing)
                strategies = {}
                for c in X.columns:
                    choice = missing_methods.get(c)
                    if choice in ("mean", "median", "most_frequent", "constant"):
                        strategies[c] = choice
                    elif pd.api.types.is_numeric_dtype(X[c]):
                        strategies[c] = "median"
                    else:
                        strategies[c] = "most_frequent"
                if X.shape[1] == 0:
                    st.error("No usable features remain. Change your cleaning or feature choices.")
                elif len(X) < 10:
                    st.error("At least 10 usable rows are recommended for training.")
                elif y.nunique() < 2:
                    st.error("Target must contain at least two distinct values.")
                else:
                    if problem == "Regression":
                        try:
                            y = pd.to_numeric(y, errors="raise")
                        except Exception:
                            st.error("Regression target must be numeric. Choose Classification or another target.")
                            st.stop()
                    stratify = None
                    if problem == "Classification":
                        counts = y.value_counts()
                        n_test = max(1, int(round(len(y) * split / 100)))
                        n_train = len(y) - n_test
                        if counts.min() < 2:
                            st.error("Each classification class needs at least two rows for a stratified split.")
                            st.stop()
                        if n_test >= y.nunique() and n_train >= y.nunique():
                            stratify = y
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=split/100, random_state=42, stratify=stratify
                        )
                        preprocessor = build_preprocessor(X_train, strategies)
                        result_rows, fitted = [], {}
                        progress = st.progress(0, text="Starting model training...")
                        status = st.empty()
                        for idx, name in enumerate(selected_model_names):
                            status.write(f"Training **{name}**...")
                            pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", clone(models[name]))])
                            try:
                                pipe.fit(X_train, y_train)
                                pred = pipe.predict(X_test)
                                metrics = metric_values(problem, y_test, pred, pipe, X_test)
                                row = {"Model": name}
                                for metric in chosen_metrics:
                                    if metric in metrics:
                                        row[metric] = metrics[metric]
                                if use_cv:
                                    try:
                                        if problem == "Classification":
                                            folds = min(cv_folds, int(y_train.value_counts().min()))
                                            scoring = "f1_weighted"
                                        else:
                                            folds = min(cv_folds, len(y_train))
                                            scoring = "neg_root_mean_squared_error"
                                        if folds >= 2:
                                            scores = cross_val_score(pipe, X_train, y_train, cv=folds, scoring=scoring, n_jobs=1)
                                            if problem == "Regression":
                                                scores = -scores
                                                row["CV RMSE"] = round(float(scores.mean()), 4)
                                            else:
                                                row["CV F1 (weighted)"] = round(float(scores.mean()), 4)
                                            row["CV Std"] = round(float(scores.std()), 4)
                                    except Exception as cv_exc:
                                        row["CV Note"] = "Skipped"
                                result_rows.append(row)
                                fitted[name] = pipe
                            except Exception as exc:
                                st.warning(f"{name} skipped: {exc}")
                            progress.progress((idx + 1) / len(selected_model_names), text=f"Processed {idx+1}/{len(selected_model_names)}")
                        progress.empty()
                        status.empty()
                        if not result_rows:
                            st.error("No model trained successfully. Check the dataset and selected features.")
                        else:
                            results = pd.DataFrame(result_rows)
                            ascending = sort_metric != "R²"
                            if sort_metric not in results.columns:
                                st.warning(f"{sort_metric} was unavailable for some models. Ranking by first available metric.")
                                sort_metric = next((m for m in metric_choices if m in results.columns), results.columns[1])
                            results = results.sort_values(sort_metric, ascending=ascending, na_position="last").reset_index(drop=True)
                            st.session_state["automl_results"] = results
                            st.session_state["automl_models"] = fitted
                            st.session_state["automl_X_train"] = X_train
                            st.session_state["automl_X_test"] = X_test
                            st.session_state["automl_y_train"] = y_train
                            st.session_state["automl_y_test"] = y_test
                            st.session_state["automl_problem"] = problem
                            st.session_state["automl_target"] = target
                            st.session_state["automl_features"] = list(X.columns)
                            st.session_state["automl_metrics_selected"] = chosen_metrics
                            st.session_state["automl_sort_metric"] = sort_metric
                            st.session_state["automl_show_cm"] = show_confusion
                            st.session_state["automl_show_cr"] = show_class_report
                            st.session_state["automl_tune"] = tune_best
                            st.session_state["automl_tune_iterations"] = tune_iterations
                            st.session_state["automl_strategies"] = strategies
                            st.session_state["automl_cleaning"] = {
                                "Original rows": len(raw_df),
                                "Rows after duplicate handling": len(raw_df.drop_duplicates()) if duplicate_action == "Remove duplicate rows" else len(raw_df),
                                "Duplicate action": duplicate_action,
                                "Missing-value choices": ", ".join(f"{k}: {v}" for k, v in missing_methods.items()) or "No missing values",
                                "Automatically removed columns": ", ".join(map(str, drop_cols_preview)) or "None",
                                "Rows removed due to missing target": target_missing_before,
                                "Fully missing features removed": ", ".join(map(str, all_missing)) or "None",
                                "Final training rows": len(X),
                                "Final feature count": X.shape[1],
                            }
                            st.session_state.pop("automl_tuned_model", None)
                            st.session_state.pop("automl_tuned_metrics", None)
                            st.success("Training complete! Open the Results & Reports tab.")
                    except Exception as exc:
                        st.error(f"Training setup failed: {exc}")

# ----------------------------- Results -----------------------------
with tab_results:
    if "automl_results" not in st.session_state:
        st.info("Train at least one model in the Model Setup tab to see results here.")
    else:
        results = st.session_state["automl_results"]
        fitted = st.session_state["automl_models"]
        problem = st.session_state["automl_problem"]
        X_train = st.session_state["automl_X_train"]
        X_test = st.session_state["automl_X_test"]
        y_train = st.session_state["automl_y_train"]
        y_test = st.session_state["automl_y_test"]
        st.subheader("Model comparison")
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download comparison CSV", results.to_csv(index=False).encode("utf-8"), "model_comparison.csv", "text/csv")
        best_name = str(results.iloc[0]["Model"])
        best_model = fitted[best_name]
        final_model = best_model
        final_name = best_name

        if st.session_state.get("automl_tune", False):
            space = tuning_space(best_name, problem)
            if space:
                if st.button("🔧 Tune leading model now"):
                    try:
                        if problem == "Classification":
                            folds = min(3, int(y_train.value_counts().min()))
                        else:
                            folds = min(3, len(y_train))
                        if folds < 2:
                            st.error("Not enough training rows/classes for tuning.")
                        else:
                            scoring = "f1_weighted" if problem == "Classification" else "neg_root_mean_squared_error"
                            search = RandomizedSearchCV(
                                clone(best_model), space,
                                n_iter=st.session_state["automl_tune_iterations"],
                                scoring=scoring, cv=folds, random_state=42, n_jobs=-1
                            )
                            with st.spinner("Tuning hyperparameters..."):
                                search.fit(X_train, y_train)
                            st.session_state["automl_tuned_model"] = search.best_estimator_
                            st.session_state["automl_tuned_params"] = search.best_params_
                            st.success("Tuning completed!")
                            st.json(search.best_params_)
                    except Exception as exc:
                        st.error(f"Tuning failed: {exc}")
            else:
                st.info("No tuning configuration is available for this model.")

        if "automl_tuned_model" in st.session_state:
            final_model = st.session_state["automl_tuned_model"]
            final_name = best_name + " (Tuned)"

        predictions = final_model.predict(X_test)
        final_metrics_all = metric_values(problem, y_test, predictions, final_model, X_test)
        final_metrics = {k: v for k, v in final_metrics_all.items() if k in st.session_state["automl_metrics_selected"] or k in ("Accuracy", "F1 (weighted)") and problem == "Classification" and not st.session_state["automl_metrics_selected"]}
        st.subheader(f"🏅 Final model: {final_name}")
        metric_cols = st.columns(min(4, max(1, len(final_metrics))))
        for i, (key, value) in enumerate(final_metrics.items()):
            metric_cols[i % len(metric_cols)].metric(key, value)

        if problem == "Classification" and st.session_state["automl_show_cm"]:
            labels = list(pd.unique(pd.concat([y_test, pd.Series(predictions)])))
            cm = confusion_matrix(y_test, predictions, labels=labels)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[str(x) for x in labels], yticklabels=[str(x) for x in labels], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        if problem == "Classification" and st.session_state["automl_show_cr"]:
            st.subheader("Detailed classification report")
            st.dataframe(pd.DataFrame(classification_report(y_test, predictions, output_dict=True, zero_division=0)).T, use_container_width=True)

        model_bytes = io.BytesIO()
        joblib.dump(final_model, model_bytes)
        model_bytes.seek(0)
        st.download_button("⬇️ Download trained model (.joblib)", model_bytes.getvalue(), "automl_best_model.joblib", "application/octet-stream")

        report_data = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_name": uploaded.name,
            "target": st.session_state["automl_target"],
            "problem_type": problem,
            "best_model": final_name,
            "final_metrics": final_metrics_all,
        }
        cleaning = st.session_state.get("automl_cleaning", {})
        pdf_bytes = pdf_report(report_data, results, cleaning)
        st.download_button("📄 Download complete PDF report", pdf_bytes, "automl_project_report.pdf", "application/pdf", type="primary")
        json_report = {**report_data, "features_used": st.session_state["automl_features"], "cleaning_summary": cleaning, "model_comparison": results.to_dict(orient="records")}
        st.download_button("⬇️ Download JSON report", json.dumps(json_report, indent=2, default=str).encode("utf-8"), "automl_report.json", "application/json")

        st.balloons()
        st.success("Your AutoML workflow is complete — report and model are ready to download.")

st.divider()
st.caption("AutoML Express | Educational project. Imputation and encoding are fitted on training data to reduce data leakage. Always validate before production use.")
