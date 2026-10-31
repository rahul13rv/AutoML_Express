import io
import json
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AutoML Platform",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AutoML Platform")
st.caption(
    "Upload a dataset, explore it, train multiple machine-learning models, "
    "compare performance, tune a model, and download your results."
)

# =========================================================
# HELPERS
# =========================================================
def load_dataset(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        # Try common CSV encodings
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Please upload a CSV or XLSX file.")


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    if not transformers:
        raise ValueError("No usable input columns were found.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_models(problem_type):
    if problem_type == "Regression":
        return {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(
                n_estimators=150, random_state=42, n_jobs=-1
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=150, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Support Vector Regression": SVR(),
        }
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(),
    }


def calculate_metrics(problem_type, y_true, y_pred):
    if problem_type == "Regression":
        mse = mean_squared_error(y_true, y_pred)
        return {
            "R2": round(float(r2_score(y_true, y_pred)), 4),
            "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "MSE": round(float(mse), 4),
            "RMSE": round(float(mse ** 0.5), 4),
        }
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision (weighted)": round(
            float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "Recall (weighted)": round(
            float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "F1 (weighted)": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
    }


def get_tuning_space(model_name, problem_type):
    if problem_type == "Regression":
        spaces = {
            "Linear Regression": {},
            "Ridge Regression": {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            "Decision Tree": {
                "model__max_depth": [None, 3, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "Random Forest": {
                "model__n_estimators": [100, 150, 250],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "Extra Trees": {
                "model__n_estimators": [100, 150, 250],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
            },
            "Gradient Boosting": {
                "model__n_estimators": [50, 100, 150],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 5],
            },
            "K-Nearest Neighbors": {
                "model__n_neighbors": [3, 5, 7, 9, 11],
                "model__weights": ["uniform", "distance"],
            },
            "Support Vector Regression": {
                "model__C": [0.1, 1, 10, 100],
                "model__kernel": ["linear", "rbf"],
                "model__epsilon": [0.05, 0.1, 0.2],
            },
        }
    else:
        spaces = {
            "Logistic Regression": {
                "model__C": [0.1, 1, 10, 100],
                "model__solver": ["lbfgs", "liblinear"],
            },
            "Decision Tree": {
                "model__max_depth": [None, 3, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "Random Forest": {
                "model__n_estimators": [100, 150, 250],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "Extra Trees": {
                "model__n_estimators": [100, 150, 250],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
            },
            "Gradient Boosting": {
                "model__n_estimators": [50, 100, 150],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 5],
            },
            "K-Nearest Neighbors": {
                "model__n_neighbors": [3, 5, 7, 9, 11],
                "model__weights": ["uniform", "distance"],
            },
            "Support Vector Classifier": {
                "model__C": [0.1, 1, 10, 100],
                "model__kernel": ["linear", "rbf"],
            },
        }
    return spaces.get(model_name, {})


def metric_score_name(problem_type):
    return "neg_root_mean_squared_error" if problem_type == "Regression" else "f1_weighted"


def make_csv_bytes(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")


# =========================================================
# DATASET UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or Excel)",
    type=["csv", "xlsx"],
)

if uploaded_file is None:
    st.info("Upload a CSV/XLSX file to begin.")
    st.stop()

try:
    df = load_dataset(uploaded_file)
except Exception as exc:
    st.error(f"Could not read this file: {exc}")
    st.stop()

if df.empty:
    st.error("This dataset is empty.")
    st.stop()

# Remove completely empty rows/columns and make duplicate column names unique
df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
if df.empty or len(df.columns) < 2:
    st.error("Please upload a dataset with rows and at least two columns.")
    st.stop()

if df.columns.duplicated().any():
    counts = {}
    new_columns = []
    for col in df.columns:
        counts[col] = counts.get(col, 0) + 1
        new_columns.append(col if counts[col] == 1 else f"{col}_{counts[col]}")
    df.columns = new_columns

st.success(f"Loaded: {uploaded_file.name}")

# =========================================================
# EDA
# =========================================================
st.header("1️⃣ Exploratory Data Analysis (EDA)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{df.shape[0]:,}")
m2.metric("Columns", df.shape[1])
m3.metric("Duplicate rows", int(df.duplicated().sum()))
m4.metric("Missing cells", int(df.isna().sum().sum()))

with st.expander("Dataset preview", expanded=True):
    preview_rows = st.slider(
        "Rows to preview", min_value=1, max_value=min(100, len(df)),
        value=min(10, len(df)), key="preview_rows"
    )
    st.dataframe(df.head(preview_rows), use_container_width=True)

with st.expander("Column information and missing values", expanded=True):
    column_info = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Values": df.isna().sum().values,
        "Missing (%)": (df.isna().mean().values * 100).round(2),
        "Unique Values": df.nunique(dropna=True).values,
    })
    st.dataframe(column_info, use_container_width=True)

with st.expander("Statistical summary"):
    st.dataframe(df.describe(include="all").T, use_container_width=True)

numeric_columns = df.select_dtypes(include="number").columns.tolist()
if numeric_columns:
    with st.expander("Numerical distribution"):
        selected_numeric = st.selectbox(
            "Select numerical column", numeric_columns, key="eda_numeric"
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(df[selected_numeric].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution: {selected_numeric}")
        st.pyplot(fig)
        plt.close(fig)

    if len(numeric_columns) >= 2:
        with st.expander("Correlation heatmap"):
            corr = df[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(10, max(5, len(numeric_columns) * 0.45)))
            sns.heatmap(corr, annot=len(numeric_columns) <= 12, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Numerical Feature Correlations")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

missing = df.isna().sum()
missing = missing[missing > 0]
if len(missing):
    with st.expander("Missing values chart"):
        fig, ax = plt.subplots(figsize=(9, 4))
        missing.sort_values(ascending=False).plot(kind="bar", ax=ax)
        ax.set_title("Missing Values per Column")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# =========================================================
# TARGET AND PROBLEM TYPE
# =========================================================
st.header("2️⃣ Choose Prediction Target")
target_column = st.selectbox("Which column should the model predict?", df.columns.tolist())

X = df.drop(columns=[target_column]).copy()
y = df[target_column].copy()

# Rows with missing target cannot be used for supervised learning.
valid_target = y.notna()
X = X.loc[valid_target].reset_index(drop=True)
y = y.loc[valid_target].reset_index(drop=True)

if X.shape[1] == 0:
    st.error("At least one feature column is required in addition to the target.")
    st.stop()

if y.nunique() < 2:
    st.error("The target column must contain at least two distinct values.")
    st.stop()

if pd.api.types.is_numeric_dtype(y):
    unique_count = y.nunique()
    suggested = "Classification" if unique_count <= 10 else "Regression"
else:
    suggested = "Classification"

st.caption(
    f"Automatic suggestion: {suggested}. Numeric targets with few distinct values "
    "can still be regression targets, so choose the correct problem type below."
)
problem_type = st.radio(
    "Problem type",
    ["Regression", "Classification"],
    index=0 if suggested == "Regression" else 1,
    horizontal=True,
)

if problem_type == "Classification" and y.nunique() < 2:
    st.error("Classification requires at least two target classes.")
    st.stop()

if problem_type == "Regression":
    try:
        y = pd.to_numeric(y, errors="raise")
    except Exception:
        st.error("Regression requires a numeric target. Choose Classification or another target.")
        st.stop()

st.write(f"**Target:** `{target_column}`")
st.write(f"**Features:** {X.shape[1]} | **Rows usable for training:** {len(X):,}")
with st.expander("Input feature preview"):
    st.dataframe(X.head(), use_container_width=True)
with st.expander("Target preview"):
    st.dataframe(y.head().to_frame(), use_container_width=True)

# Remove features that are entirely missing; imputers cannot learn from these.
all_missing_columns = X.columns[X.isna().all()].tolist()
if all_missing_columns:
    st.warning(
        "These feature columns contain only missing values and will be removed: "
        + ", ".join(map(str, all_missing_columns))
    )
    X = X.drop(columns=all_missing_columns)

if X.shape[1] == 0:
    st.error("No usable feature columns remain after removing all-missing columns.")
    st.stop()

# Convert datetime features into string representation for compatibility.
for col in X.columns:
    if pd.api.types.is_datetime64_any_dtype(X[col]):
        X[col] = X[col].astype(str)

# Remove infinite numeric values by treating them as missing
X = X.replace([float("inf"), float("-inf")], pd.NA)

# =========================================================
# TRAINING OPTIONS
# =========================================================
st.header("3️⃣ Training Settings")
left, right = st.columns(2)
with left:
    test_size = st.slider("Test data (%)", min_value=10, max_value=40, value=20, step=5)
with right:
    cv_folds = st.selectbox("Cross-validation folds", [3, 5, 10], index=0)

available_models = get_models(problem_type)
model_names = list(available_models.keys())
selected_models = st.multiselect(
    "Choose models to train (select multiple)",
    model_names,
    default=model_names,
)
run_training = st.button("🚀 Train and Compare Models", type="primary")

if run_training:
    if not selected_models:
        st.error("Select at least one model.")
        st.stop()

    if len(X) < 10:
        st.error("Please upload at least 10 usable rows for a meaningful train/test split.")
        st.stop()

    stratify_values = None
    if problem_type == "Classification":
        class_counts = y.value_counts()
        if class_counts.min() < 2:
            st.error(
                "At least one target class has fewer than 2 rows. "
                "Add more examples for each class or choose a different target."
            )
            st.stop()
        # Stratify only if every class can appear in both train and test
        n_classes = y.nunique()
        n_test = max(1, int(round(len(y) * test_size / 100)))
        n_train = len(y) - n_test
        if class_counts.min() >= 2 and n_test >= n_classes and n_train >= n_classes:
            stratify_values = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size / 100,
            random_state=42,
            stratify=stratify_values,
        )
    except Exception as exc:
        st.error(f"Could not split the dataset: {exc}")
        st.stop()

    if problem_type == "Classification":
        train_class_counts = y_train.value_counts()
        effective_cv = min(cv_folds, int(train_class_counts.min()))
        if effective_cv < 2:
            st.warning("Cross-validation skipped because a training class has too few samples.")
            effective_cv = None
    else:
        effective_cv = min(cv_folds, len(X_train))
        if effective_cv < 2:
            effective_cv = None

    preprocessor = build_preprocessor(X_train)
    results = []
    fitted_models = {}
    progress = st.progress(0, text="Training models...")

    for i, model_name in enumerate(selected_models):
        estimator = available_models[model_name]
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", clone(estimator)),
            ]
        )
        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            metrics = calculate_metrics(problem_type, y_test, predictions)

            cv_score = None
            cv_std = None
            if effective_cv:
                try:
                    scores = cross_val_score(
                        pipeline,
                        X_train,
                        y_train,
                        cv=effective_cv,
                        scoring=metric_score_name(problem_type),
                        n_jobs=1,
                    )
                    # Convert negative RMSE to positive for easier interpretation
                    if problem_type == "Regression":
                        scores = -scores
                    cv_score = float(scores.mean())
                    cv_std = float(scores.std())
                except Exception:
                    cv_score = None
                    cv_std = None

            row = {"Model": model_name, **metrics}
            if cv_score is not None:
                label = "CV RMSE" if problem_type == "Regression" else "CV F1 (weighted)"
                row[label] = round(cv_score, 4)
                row["CV Std"] = round(cv_std, 4)
            results.append(row)
            fitted_models[model_name] = pipeline

        except Exception as exc:
            st.warning(f"{model_name} could not be trained: {exc}")

        progress.progress((i + 1) / len(selected_models), text=f"Processed {i + 1}/{len(selected_models)} models")

    progress.empty()

    if not results:
        st.error("No models trained successfully. Check your data and target column.")
        st.stop()

    results_df = pd.DataFrame(results)

    if problem_type == "Regression":
        sort_column = "RMSE"
        results_df = results_df.sort_values(sort_column, ascending=True)
    else:
        sort_column = "F1 (weighted)"
        results_df = results_df.sort_values(sort_column, ascending=False)

    st.session_state["automl_results"] = results_df
    st.session_state["automl_fitted_models"] = fitted_models
    st.session_state["automl_X_train"] = X_train
    st.session_state["automl_X_test"] = X_test
    st.session_state["automl_y_train"] = y_train
    st.session_state["automl_y_test"] = y_test
    st.session_state["automl_problem_type"] = problem_type
    st.session_state["automl_target"] = target_column
    st.session_state["automl_selected_models"] = selected_models
    st.session_state["automl_cv"] = effective_cv
    st.session_state["automl_preprocessor"] = preprocessor
    st.success("Model training and comparison completed!")

# =========================================================
# DISPLAY RESULTS AND TUNING
# =========================================================
if "automl_results" in st.session_state:
    results_df = st.session_state["automl_results"]
    fitted_models = st.session_state["automl_fitted_models"]
    saved_problem_type = st.session_state["automl_problem_type"]
    X_train = st.session_state["automl_X_train"]
    X_test = st.session_state["automl_X_test"]
    y_train = st.session_state["automl_y_train"]
    y_test = st.session_state["automl_y_test"]

    # Avoid showing results from a prior dataset/target/problem selection.
    # Results persist across Streamlit reruns, but user can retrain with the button.
    st.header("4️⃣ Model Comparison")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download model comparison (CSV)",
        data=make_csv_bytes(results_df),
        file_name="automl_model_comparison.csv",
        mime="text/csv",
    )

    if saved_problem_type == "Regression":
        best_name = str(results_df.iloc[0]["Model"])
    else:
        best_name = str(results_df.iloc[0]["Model"])

    st.subheader("🏆 Best Model (before tuning)")
    st.write(f"Based on the selected test metric, the leading model is **{best_name}**.")
    best_pipeline = fitted_models[best_name]

    st.header("5️⃣ Hyperparameter Tuning")
    st.write(
        "Tuning tries different model settings using training data only. "
        "The test set is kept aside for final evaluation."
    )
    tuning_space = get_tuning_space(best_name, saved_problem_type)
    if tuning_space:
        tune_now = st.checkbox("Tune the leading model", value=False)
        if tune_now:
            tune_iter = st.slider("Randomized search iterations", 5, 30, 10)
            tuning_cv = st.selectbox("Tuning CV folds", [3, 5], index=0, key="tuning_cv")

            if st.button("🔧 Run Hyperparameter Tuning"):
                try:
                    if saved_problem_type == "Classification":
                        min_class_count = int(y_train.value_counts().min())
                        actual_tuning_cv = min(tuning_cv, min_class_count)
                        if actual_tuning_cv < 2:
                            st.error("Not enough examples per class in training data for tuning CV.")
                            st.stop()
                    else:
                        actual_tuning_cv = min(tuning_cv, len(X_train))
                        if actual_tuning_cv < 2:
                            st.error("Not enough training rows for tuning CV.")
                            st.stop()

                    search = RandomizedSearchCV(
                        estimator=clone(best_pipeline),
                        param_distributions=tuning_space,
                        n_iter=tune_iter,
                        scoring=metric_score_name(saved_problem_type),
                        cv=actual_tuning_cv,
                        random_state=42,
                        n_jobs=-1,
                        error_score="raise",
                    )
                    with st.spinner("Tuning model. This may take a few minutes..."):
                        search.fit(X_train, y_train)

                    tuned_model = search.best_estimator_
                    tuned_predictions = tuned_model.predict(X_test)
                    tuned_metrics = calculate_metrics(
                        saved_problem_type, y_test, tuned_predictions
                    )

                    st.session_state["automl_tuned_model"] = tuned_model
                    st.session_state["automl_tuned_name"] = best_name + " (Tuned)"
                    st.session_state["automl_tuned_metrics"] = tuned_metrics
                    st.session_state["automl_tuned_params"] = search.best_params_

                    st.success("Hyperparameter tuning completed!")
                    st.write("**Best parameters:**")
                    st.json(search.best_params_)
                    st.write("**Tuned model test metrics:**")
                    st.dataframe(
                        pd.DataFrame([tuned_metrics]),
                        use_container_width=True,
                        hide_index=True,
                    )
                except Exception as exc:
                    st.error(f"Tuning failed: {exc}")
    else:
        st.info("No tuning search space is configured for this model.")

    # Tuned model is used for final artifacts if available
    final_model = st.session_state.get("automl_tuned_model", best_pipeline)
    final_name = st.session_state.get("automl_tuned_name", best_name)
    final_predictions = final_model.predict(X_test)
    final_metrics = calculate_metrics(saved_problem_type, y_test, final_predictions)

    st.header("6️⃣ Final Model Evaluation")
    st.write(f"**Model:** {final_name}")
    st.dataframe(
        pd.DataFrame([final_metrics]),
        use_container_width=True,
        hide_index=True,
    )

    if saved_problem_type == "Classification":
        labels = list(pd.unique(pd.concat([y_test, pd.Series(final_predictions)])))
        cm = confusion_matrix(y_test, final_predictions, labels=labels)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[str(x) for x in labels],
            yticklabels=[str(x) for x in labels],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        report_dict = classification_report(
            y_test, final_predictions, output_dict=True, zero_division=0
        )
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(report_dict).T, use_container_width=True)

    # Save serialized model in memory for download
    model_buffer = io.BytesIO()
    joblib.dump(final_model, model_buffer)
    model_buffer.seek(0)
    st.download_button(
        "⬇️ Download trained model (.joblib)",
        data=model_buffer.getvalue(),
        file_name="automl_best_model.joblib",
        mime="application/octet-stream",
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_column": st.session_state["automl_target"],
        "problem_type": saved_problem_type,
        "selected_model": final_name,
        "metrics": final_metrics,
        "dataset_rows_used": int(len(X_train) + len(X_test)),
        "number_of_features": int(X_train.shape[1]),
    }
    st.download_button(
        "⬇️ Download evaluation report (JSON)",
        data=json.dumps(report, indent=2, default=str).encode("utf-8"),
        file_name="automl_evaluation_report.json",
        mime="application/json",
    )

    st.caption(
        "Note: the downloaded model is a scikit-learn pipeline that includes preprocessing. "
        "Only load model files from sources you trust."
    )

st.divider()
st.caption("AutoML Platform • Educational project. Always validate models before production use.")
