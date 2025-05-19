# EDA Packages
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import re
import streamlit as st
from PIL import Image

# Model Imports (centralized)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge, Lasso # Note: These are regression models; used here heuristically for classification.

# Page Configuration
try:
    im = Image.open("favicon.ico")
    st.set_page_config(
        page_title="Sundar-Bot AutoML",
        page_icon=im,
        layout="wide",
    )
except FileNotFoundError:
    st.set_page_config(
        page_title="Sundar-Bot AutoML",
        layout="wide",
    )
    st.warning("favicon.ico not found. Using default favicon.")

st.title("AutoModeler Sundar Version")
st.header("For Classification Tasks (Categorical Target)")

@st.cache_data # Replaced st.cache with st.cache_data
def run_automl_pipeline(df_original: pd.DataFrame, target_col_name: str, sort_metric: str) -> pd.DataFrame:
    """
    Runs the automated machine learning pipeline.
    Processes data, trains multiple models, evaluates them, and returns a summary.
    """
    df = df_original.copy() # Work on a copy

    # --- 1. Preprocessing ---
    # Sanitize column names (including the target column name passed as argument)
    df = df.rename(columns=lambda x: re.sub(r'[^A-Za-z0-9_]+', '', str(x)))
    target_col_name = re.sub(r'[^A-Za-z0-9_]+', '', str(target_col_name))

    if target_col_name not in df.columns:
        st.error(f"Target column '{target_col_name}' not found in the DataFrame after sanitization. Available columns: {df.columns.tolist()}")
        return pd.DataFrame()

    # Impute missing values
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mode()[0])
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else: # For other types, fill with a common placeholder or mode
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")


    # Label encode all object/categorical columns (including target if it's object type)
    # Store encoders in case they are needed later (e.g., for inverse transform)
    label_encoders = {}
    for col in df.columns: # Iterate over all columns to catch any that might be object type
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            le = LabelEncoder()
            # Handle potential new values in a column if it was partially numeric then filled with strings
            df[col] = df[col].astype(str) 
            df[col] = pd.Series(le.fit_transform(df[col]), index=df.index)
            label_encoders[col] = le
    
    if target_col_name not in df.columns: # Re-check, as dtype changes might affect column access
        st.error(f"Target column '{target_col_name}' was lost during preprocessing. This can happen if it was fully NaN and dropped or due to type issues.")
        return pd.DataFrame()

    # --- 2. Data Splitting ---
    X = df.drop(columns=[target_col_name])
    y = df[target_col_name]

    if y.nunique() < 2:
        st.warning(f"The target column '{target_col_name}' has {y.nunique()} unique value(s). "
                   "At least 2 unique values are required for classification. Cannot proceed with model training.")
        return pd.DataFrame()

    try:
        # Stratify if possible, especially for imbalanced datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y if y.nunique() > 1 else None
        )
    except ValueError as e:
        st.error(f"Error during train-test split (possibly due to stratification issues with small classes): {e}")
        st.info(f"Target column unique values: {y.nunique()}, counts: {y.value_counts().to_dict()}")
        st.info("Attempting split without stratification...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        except Exception as e_nostrat:
            st.error(f"Train-test split failed even without stratification: {e_nostrat}")
            return pd.DataFrame()


    # --- 3. Model Training and Evaluation ---
    models_to_evaluate = {
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "LGBMClassifier": LGBMClassifier(random_state=42, verbosity=-1),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        "AdaBoostClassifier": AdaBoostClassifier(n_estimators=50, random_state=42),
        "GaussianNB": GaussianNB(),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=min(5, len(X_train)) if len(X_train) > 0 else 1), # Adjusted n_neighbors
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
        "RidgeClassifierHeuristic": Ridge(random_state=42), # Heuristic for classification
        "LassoClassifierHeuristic": Lasso(random_state=42)  # Heuristic for classification
    }

    model_results_list = []
    avg_method = 'binary' if y.nunique() == 2 else 'weighted' # For precision, recall, F1

    for model_name, model_instance in models_to_evaluate.items():
        try:
            st.write(f"Training {model_name}...")
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)

            # For Ridge/Lasso, predictions are continuous; round them for classification.
            if "Heuristic" in model_name:
                y_pred = np.round(y_pred).astype(int)
                min_class, max_class = y_test.min(), y_test.max()
                y_pred = np.clip(y_pred, min_class, max_class) # Ensure predictions are valid classes

            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred, average=avg_method, zero_division=0)
            recall = metrics.recall_score(y_test, y_pred, average=avg_method, zero_division=0)
            f1 = metrics.f1_score(y_test, y_pred, average=avg_method, zero_division=0)
            
            roc_auc = np.nan # Default to NaN
            try:
                if hasattr(model_instance, "predict_proba"):
                    y_proba = model_instance.predict_proba(X_test)
                    if y.nunique() == 2: # Binary classification
                        roc_auc = metrics.roc_auc_score(y_test, y_proba[:, 1])
                    else: # Multiclass classification
                        roc_auc = metrics.roc_auc_score(y_test, y_proba, multi_class='ovr', average=avg_method)
                elif "Heuristic" in model_name and y.nunique() == 2 : # For Ridge/Lasso in binary case, allow AUC on rounded preds
                     roc_auc = metrics.roc_auc_score(y_test, y_pred)
                else:
                    st.caption(f"ROC AUC not computed for {model_name} as predict_proba is unavailable or it's a multiclass heuristic.")
            except ValueError as e_roc:
                st.caption(f"Could not calculate ROC AUC for {model_name} (ValueError: {e_roc}).")
            except Exception as e_roc_gen:
                st.caption(f"Could not calculate ROC AUC for {model_name} (Error: {e_roc_gen}).")

            model_results_list.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "AUC-ROC": roc_auc,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            })
            st.write(f"✅ {model_name} evaluated.")
        except Exception as e:
            st.warning(f"⚠️ Could not train or evaluate {model_name}. Error: {e}")
            model_results_list.append({
                "Model": model_name, "Accuracy": np.nan, "AUC-ROC": np.nan,
                "Precision": np.nan, "Recall": np.nan, "F1": np.nan
            })

    results_df = pd.DataFrame(model_results_list)
    if not results_df.empty:
        if sort_metric in results_df.columns and not results_df[sort_metric].isnull().all():
            results_df = results_df.sort_values(by=sort_metric, ascending=False).reset_index(drop=True)
        elif sort_metric in results_df.columns:
            st.warning(f"Cannot sort by '{sort_metric}' as all its values are NaN. Displaying results unsorted.")
        else: # Should not happen if sort_metric is from the predefined list
            st.warning(f"Sort metric '{sort_metric}' not found in results. Displaying unsorted results.")
    
    return results_df

# --- Streamlit UI ---

# Initialize session state variables
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'target_column_name' not in st.session_state:
    st.session_state.target_column_name = ""
if 'sort_metric_selection' not in st.session_state:
    st.session_state.sort_metric_selection = "Accuracy" # Default sort metric

uploaded_file = st.file_uploader(
    label="Upload a CSV file for classification",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df_temp = pd.read_csv(uploaded_file)
        # Sanitize column names of the loaded dataframe right away
        original_cols = df_temp.columns.tolist()
        sanitized_cols_map = {col: re.sub(r'[^A-Za-z0-9_]+', '', str(col)) for col in original_cols}
        df_temp.rename(columns=sanitized_cols_map, inplace=True)
        
        st.session_state.df_loaded = df_temp
        st.session_state.original_sanitized_map = sanitized_cols_map # Store mapping if needed later
        
        st.success("File uploaded and column names sanitized successfully!")
        st.write("Preview of the first 5 rows (with sanitized column names):")
        st.dataframe(st.session_state.df_loaded.head())
    except Exception as e:
        st.error(f"Error reading or processing CSV file: {e}")
        st.session_state.df_loaded = None # Reset if error
else:
    if st.session_state.df_loaded is None: # Only show if no df is loaded yet
        st.info("Please upload a CSV file to begin.")


if st.session_state.df_loaded is not None:
    sanitized_df_cols = st.session_state.df_loaded.columns.tolist()
    
    # Use a selectbox for target column for better UX with sanitized names
    st.session_state.target_column_name = st.selectbox(
        "Select the target column:",
        options=sanitized_df_cols,
        index=0 if sanitized_df_cols else 0, # Default to first col
        key="target_select" # Add a key for robust state management
    )
    
    st.session_state.sort_metric_selection = st.sidebar.selectbox(
        "Pick the metric to sort results by:",
        options=("Accuracy", "AUC-ROC", "Precision", "Recall", "F1"), # Corrected "Precission"
        index=0, # Default to Accuracy
        key="metric_select"
    )

    if st.button("🚀 Run AutoML Pipeline", key="run_button"):
        if not st.session_state.target_column_name:
            st.warning("Please select the target column.")
        elif st.session_state.target_column_name not in sanitized_df_cols:
            # This case should be rare with selectbox, but good for robustness
            st.error(f"Target column '{st.session_state.target_column_name}' is not valid.")
        else:
            st.info(f"Target column: '{st.session_state.target_column_name}'")
            st.info(f"Sorting results by: '{st.session_state.sort_metric_selection}'")
            with st.spinner("🤖 The AutoML magic is happening... Please wait."):
                # Pass a copy of the dataframe to the cached function
                results = run_automl_pipeline(
                    st.session_state.df_loaded.copy(), 
                    st.session_state.target_column_name, 
                    st.session_state.sort_metric_selection
                )
                st.session_state.results_df = results
            
            if st.session_state.results_df is not None and not st.session_state.results_df.empty:
                st.success("🎉 Processing Complete!")
                st.write("📊 Model Performance Results:")
                st.dataframe(st.session_state.results_df.style.format({
                    "Accuracy": "{:.4f}", "AUC-ROC": "{:.4f}", 
                    "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"
                }))
            elif st.session_state.results_df is not None and st.session_state.results_df.empty:
                 st.info("Processing finished, but no valid model results were generated. Please check the data and target column selection, and review any warnings above.")
            # If results_df is None, errors would have been shown in the function.

# To show results if they are already computed and user interacts with something else (e.g. sidebar)
# This might be redundant if the button is the only trigger for computation.
# However, if we want results to persist display across minor non-button interactions:
elif st.session_state.results_df is not None and not st.session_state.results_df.empty:
    st.write("📊 Previously Computed Model Performance Results:")
    st.dataframe(st.session_state.results_df.style.format({
        "Accuracy": "{:.4f}", "AUC-ROC": "{:.4f}", 
        "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"
    }))

