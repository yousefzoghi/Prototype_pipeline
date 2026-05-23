"""
Classification — Train and evaluate machine learning models (SVM, Random Forest, XGBoost, etc.) on extracted EEG features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef, precision_recall_curve, auc, roc_curve, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from utils.sidebar import render_sidebar, render_sidebar_footer
from utils.signal_processing import DEFAULT_MAPPING

st.set_page_config(page_title="Classification", layout="wide")
render_sidebar()
render_sidebar_footer()

st.title("🧠 EEG Machine Learning Classification")
st.markdown("""
This page allows you to train and evaluate machine learning models (**SVM**, **Random Forest**, **XGBoost**, **LDA**, **MLP**, or **k-NN**) on the feature dataset extracted in the Preprocessing page.
""")

# ── 1. Data Loading ────────────────────────────────────────────────────────
st.divider()
st.header("1. Load Dataset")

df = None

# Option A: From Session State (Internal)
feat_info = st.session_state.get("extracted_features")
if feat_info and "epoch_tags" in feat_info:
    st.success("✅ Found extracted features in memory from Preprocessing page.")
    
    # Reconstruct DataFrame (Wide Format)
    f3d = feat_info["data"]
    tags = feat_info["epoch_tags"]
    ch_names = feat_info["ch_names"]
    f_names = feat_info["feature_names"]
    
    n_ep, n_ch, n_feat = f3d.shape
    all_rows = []
    for e in range(n_ep):
        row = []
        for c in range(n_ch):
            row.extend(list(f3d[e, c, :]))
        row.append(tags[e])
        all_rows.append(row)
        
    col_names = []
    for ch in ch_names:
        ch_label = DEFAULT_MAPPING.get(ch, ch)
        for fn in f_names:
            col_names.append(f"{ch_label}_{fn}")
    col_names.append("Tag")
    
    df = pd.DataFrame(all_rows, columns=col_names)
else:
    st.info("No features found in session. Please run extraction in the Preprocessing page or upload a CSV below.")

# Option B: File Upload (External)
uploaded_file = st.file_uploader("Or upload an exported features CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Uploaded external CSV successfully.")

if df is None:
    st.warning("Please provide a dataset to continue.")
    st.stop()

# ── 2. Data Preparation ────────────────────────────────────────────────────
st.divider()
st.header("2. Data Preparation")

st.write("**Dataset Preview:**")
st.dataframe(df.head(10), use_container_width=True)

# Data Cleaning Options
dccol1, dccol2 = st.columns(2)
with dccol1:
    drop_none = st.checkbox("Drop 'none' tags", value=True, help="Remove epochs that don't have a specific label.")
    drop_nans = st.checkbox("Drop rows with NaN", value=True, help="Remove epochs with missing feature values.")
    under_sample = st.checkbox("Random Under-sampling", value=False, help="Downsample the majority class to match the minority class size. Useful for heavily imbalanced data.")
with dccol2:
    balance_check = st.checkbox("Show class distribution", value=True)

if drop_none and "Tag" in df.columns:
    df = df[df["Tag"] != "none"].reset_index(drop=True)

if under_sample and "Tag" in df.columns:
    counts = df["Tag"].value_counts()
    if len(counts) > 1:
        min_size = counts.min()
        df = df.groupby("Tag").apply(lambda x: x.sample(min_size, random_state=42)).reset_index(drop=True)
        st.info(f"Balanced dataset by under-sampling to {min_size} samples per class.")

if drop_nans:
    # Replace inf with nan so they get dropped too
    df = df.replace([np.inf, -np.inf], np.nan)
    nan_rows = df.isna().any(axis=1).sum()
    if nan_rows > 0:
        st.warning(f"⚠️ Found {nan_rows} rows with NaN/Inf values. Dropping them to ensure model compatibility.")
        df = df.dropna().reset_index(drop=True)

if balance_check and "Tag" in df.columns:
    counts = df["Tag"].value_counts()
    fig_balance = px.bar(x=counts.index, y=counts.values, labels={'x': 'Class', 'y': 'Count'}, 
                         title="Class Distribution", template="plotly_dark", color=counts.index)
    st.plotly_chart(fig_balance, use_container_width=True)

if len(df) < 5:
    st.error("Dataset too small for training. Need at least 5 labeled samples.")
    st.stop()

# ── 3. Model Training ──────────────────────────────────────────────────────
st.divider()
st.header("3. Train Model")

# Features & Target
X = df.drop(columns=["Tag"])
y = df["Tag"]

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Algorithm Selection
model_type = st.radio("Select Classifier", ["Support Vector Machine (SVM)", "Random Forest", "XGBoost", "Linear Discriminant Analysis (LDA)", "Multi-Layer Perceptron (MLP)", "k-Nearest Neighbors (k-NN)"], horizontal=True)

# Tuning Option
st.divider()
st.subheader("Model Optimizations")
ocol1, ocol2, ocol3 = st.columns(3)
with ocol1:
    handle_imbalance = st.checkbox("⚖️ Handle Class Imbalance", value=True, help="Uses 'balanced' class weights or sample weighting to penalize minority class errors more heavily.")
    opt_metric = st.selectbox("Optimization Metric", ["balanced_accuracy", "f1_weighted", "average_precision", "accuracy"], index=0, help="Metric used by Grid Search to select the best model. 'average_precision' (AUPRC) is best for imbalanced data.")
with ocol2:
    use_grid_search = st.checkbox("🔍 Enable Grid Search", value=False, help="Test multiple configurations and select the one with the highest accuracy.")
with ocol3:
    custom_threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5, 0.05, help="Only for binary classification (or first non-majority class). Lowering this increases Recall for the minority class.")

# Hyperparameters
st.write("**Manual Parameters (Ignored if Grid Search is enabled)**")
hcol1, hcol2, hcol3 = st.columns(3)

with hcol1:
    test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100

if model_type == "Support Vector Machine (SVM)":
    with hcol2:
        svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
    with hcol3:
        svm_c = st.number_input("C (Regularization)", value=1.0, min_value=0.01, max_value=100.0, step=0.1)
elif model_type == "Random Forest":
    with hcol2:
        rf_trees = st.number_input("Number of Trees", value=100, min_value=10, max_value=1000, step=10)
    with hcol3:
        rf_depth = st.number_input("Max Depth", value=10, min_value=1, max_value=100, step=1)
elif model_type == "XGBoost":
    with hcol2:
        xgb_eta = st.number_input("Learning Rate (eta)", value=0.3, min_value=0.01, max_value=1.0, step=0.05)
    with hcol3:
        xgb_depth = st.number_input("Max Depth", value=6, min_value=1, max_value=20, step=1)
elif model_type == "Linear Discriminant Analysis (LDA)":
    with hcol2:
        lda_solver = st.selectbox("Solver", ["svd", "lsqr", "eigen"], index=0)
    with hcol3:
        if lda_solver in ["lsqr", "eigen"]:
            lda_shrinkage = st.selectbox("Shrinkage", [None, "auto"], index=1)
        else:
            st.info("LDA is highly efficient for EEG.")
            lda_shrinkage = None
elif model_type == "Multi-Layer Perceptron (MLP)":
    with hcol2:
        mlp_layers = st.text_input("Hidden Layers (e.g. 100,50)", value="100,50")
        mlp_layers = tuple(map(int, mlp_layers.split(",")))
    with hcol3:
        mlp_max_iter = st.number_input("Max Iterations", value=500, min_value=10, max_value=5000, step=10)
else: # k-NN
    with hcol2:
        knn_k = st.number_input("Number of Neighbors (k)", value=5, min_value=1, max_value=50, step=1)
    with hcol3:
        knn_weights = st.selectbox("Weights", ["uniform", "distance"], index=0)

if st.button("🚀 Train Model", type="primary", use_container_width=True):
    with st.spinner("Training model..."):
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Final safety check for NaNs/Infs
        if np.any(np.isnan(X_train_scaled)) or np.any(np.isinf(X_train_scaled)):
            st.error("💥 Dataset contains NaN or Inf values after scaling. Please ensure 'Drop rows with NaN' is enabled and your data is valid.")
            st.stop()
        
        # Class Weights
        cw = 'balanced' if handle_imbalance else None
        train_weights = compute_sample_weight(class_weight='balanced', y=y_train) if handle_imbalance else None

        best_params = None
        
        if use_grid_search:
            # Define Grids
            if model_type == "Support Vector Machine (SVM)":
                model_base = SVC(probability=True, random_state=42, class_weight=cw)
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            elif model_type == "Random Forest":
                model_base = RandomForestClassifier(random_state=42, class_weight=cw)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            elif model_type == "XGBoost":
                model_base = XGBClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 6, 10]
                }
            elif model_type == "Linear Discriminant Analysis (LDA)":
                model_base = LinearDiscriminantAnalysis()
                param_grid = [
                    {'solver': ['svd']},
                    {'solver': ['lsqr', 'eigen'], 'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]}
                ]
            elif model_type == "Multi-Layer Perceptron (MLP)":
                model_base = MLPClassifier(max_iter=1000, random_state=42)
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 25)],
                    'activation': ['tanh', 'relu'],
                    'alpha': [0.0001, 0.05],
                    'learning_rate': ['constant', 'adaptive'],
                }
            else: # k-NN
                model_base = KNeighborsClassifier()
                param_grid = {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            
            # Run Grid Search
            cv_folds = max(2, min(5, len(X_train)//5))
            grid = GridSearchCV(model_base, param_grid, cv=cv_folds, scoring=opt_metric, n_jobs=-1)
            
            if model_type == "XGBoost" and handle_imbalance:
                grid.fit(X_train_scaled, y_train, sample_weight=train_weights)
            else:
                grid.fit(X_train_scaled, y_train)
            
            model = grid.best_estimator_
            best_params = grid.best_params_
        else:
            # Initialize Model Manually
            if model_type == "Support Vector Machine (SVM)":
                model = SVC(kernel=svm_kernel, C=svm_c, probability=True, random_state=42, class_weight=cw)
            elif model_type == "Random Forest":
                model = RandomForestClassifier(n_estimators=rf_trees, max_depth=rf_depth, random_state=42, class_weight=cw)
            elif model_type == "XGBoost":
                model = XGBClassifier(learning_rate=xgb_eta, max_depth=xgb_depth, random_state=42)
            elif model_type == "Linear Discriminant Analysis (LDA)":
                model = LinearDiscriminantAnalysis(solver=lda_solver, shrinkage=lda_shrinkage)
            elif model_type == "Multi-Layer Perceptron (MLP)":
                model = MLPClassifier(hidden_layer_sizes=mlp_layers, max_iter=mlp_max_iter, random_state=42)
            else: # k-NN
                model = KNeighborsClassifier(n_neighbors=knn_k, weights=knn_weights)
            
            if model_type == "XGBoost" and handle_imbalance:
                model.fit(X_train_scaled, y_train, sample_weight=train_weights)
            else:
                model.fit(X_train_scaled, y_train)
        
        # Predict with Threshold Tuning (Binary Only)
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test_scaled)
            if len(class_names) == 2:
                y_pred = (y_probs[:, 1] >= custom_threshold).astype(int)
            else:
                y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test_scaled)
            y_probs = None
        
        # ── 4. Evaluation ────────────────────────────────────────────────────
        st.divider()
        st.header("4. Evaluation Results")
        
        if best_params:
            st.success(f"🎯 **Best Parameters Found (Optimized for {opt_metric}):** {best_params}")
        
        ecol1, ecol2, ecol3, ecol4 = st.columns(4)
        acc = accuracy_score(y_test, y_pred)
        b_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)
        
        ecol1.metric("Overall Accuracy", f"{acc:.2%}")
        ecol2.metric("Balanced Accuracy", f"{b_acc:.2%}")
        ecol3.metric("F1-Score (Weighted)", f"{f1:.2f}")
        ecol4.metric("MCC Score", f"{mcc:.2f}")
        
        # ROC & PR Curves
        if y_probs is not None:
            st.divider()
            st.subheader("Performance Curves")
            ccol1, ccol2 = st.columns(2)
            
            with ccol1:
                # 📊 Unified ROC Curve (All classes)
                fig_roc = go.Figure()
                for i in range(len(class_names)):
                    fpr, tpr, _ = roc_curve(y_test == i, y_probs[:, i])
                    class_auc = auc(fpr, tpr)
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{class_names[i]} (AUC={class_auc:.2f})", mode='lines'))
                
                fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                fig_roc.update_layout(title="ROC Curves (One-vs-Rest)", xaxis_title="False Positive Rate", 
                                      yaxis_title="True Positive Rate", template="plotly_dark",
                                      legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99))
                st.plotly_chart(fig_roc, use_container_width=True)

            with ccol2:
                # 📈 Precision-Recall Curve (All classes)
                fig_pr = go.Figure()
                for i in range(len(class_names)):
                    prec, rec, _ = precision_recall_curve(y_test == i, y_probs[:, i])
                    class_auprc = auc(rec, prec)
                    fig_pr.add_trace(go.Scatter(x=rec, y=prec, name=f"{class_names[i]} (AUC={class_auprc:.2f})", mode='lines'))
                
                fig_pr.update_layout(title="Precision-Recall Curves (One-vs-Rest)", xaxis_title="Recall", 
                                     yaxis_title="Precision", template="plotly_dark",
                                     legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99))
                st.plotly_chart(fig_pr, use_container_width=True)
        
        # ── Error Analysis ────────────────────────────────────────────────
        if "auditory" in class_names:
            st.divider()
            st.subheader("🕵️ Error Analysis: 'auditory' class")
            aud_idx = list(class_names).index("auditory")
            
            # Identify False Positives for 'auditory'
            # (Predicted 'auditory', but actually something else)
            fps_idx = np.where((y_pred == aud_idx) & (y_test != aud_idx))[0]
            fns_idx = np.where((y_pred != aud_idx) & (y_test == aud_idx))[0]
            
            st.write(f"Found **{len(fps_idx)}** False Positives and **{len(fns_idx)}** False Negatives for the 'auditory' class.")
            
            if len(fps_idx) > 0:
                with st.expander("View False Positive Samples (Mistaken for auditory)"):
                    fp_data = X_test.iloc[fps_idx].copy()
                    fp_data["Actual_Label"] = le.inverse_transform(y_test[fps_idx])
                    if y_probs is not None:
                        fp_data["Auditory_Prob"] = y_probs[fps_idx, aud_idx]
                    st.dataframe(fp_data, use_container_width=True)
            
            if len(fns_idx) > 0:
                with st.expander("View False Negative Samples (Missed auditory events)"):
                    fn_data = X_test.iloc[fns_idx].copy()
                    fn_data["Predicted_As"] = le.inverse_transform(y_pred[fns_idx])
                    if y_probs is not None:
                        fn_data["Auditory_Prob"] = y_probs[fns_idx, aud_idx]
                    st.dataframe(fn_data, use_container_width=True)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, 
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=class_names, y=class_names,
                           title=f"Confusion Matrix ({model_type})",
                           color_continuous_scale='Viridis',
                           template="plotly_dark")
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Detailed Report
        with st.expander("Show Detailed Classification Report"):
            report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
            st.dataframe(pd.DataFrame(report_dict).transpose(), use_container_width=True)
            
        # Store model in session state
        st.session_state["trained_classifier"] = {
            "model": model,
            "scaler": scaler,
            "le": le,
            "accuracy": acc,
            "type": model_type,
            "kernel": svm_kernel if model_type == "Support Vector Machine (SVM)" else None
        }
        st.success("Model trained and evaluated successfully!")

# ── 5. Feature Insights ───────────────────────────────────────────────────
if "trained_classifier" in st.session_state:
    st.divider()
    st.header("5. Feature Insights")
    
    train_info = st.session_state["trained_classifier"]
    model = train_info["model"]
    m_type = train_info["type"]
    
    importance_df = None
    
    if m_type == "Random Forest":
        # Native RF Importance
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        title = "Random Forest: Feature Importance (Gini)"
    
    elif m_type == "XGBoost":
        # Native XGBoost Importance
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        title = "XGBoost: Feature Importance (Gain)"
        
    elif m_type == "Support Vector Machine (SVM)" and train_info["kernel"] == "linear":
        # Linear SVM Coefficients
        coefs = np.abs(model.coef_)
        avg_importance = np.mean(coefs, axis=0)
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": avg_importance
        }).sort_values(by="Importance", ascending=False)
        title = "Linear SVM: Feature Importance (Mean |Coeff|)"

    elif m_type == "Linear Discriminant Analysis (LDA)":
        # LDA Coefficients
        coefs = np.abs(model.coef_)
        avg_importance = np.mean(coefs, axis=0)
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": avg_importance
        }).sort_values(by="Importance", ascending=False)
        title = "LDA: Feature Importance (Mean |Coeff|)"

    if importance_df is not None:
        fig_imp = px.bar(importance_df.head(20), x="Importance", y="Feature", orientation='h',
                         title=title, template="plotly_dark", color="Importance")
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance insights are available for Random Forest, XGBoost, and Linear SVM kernels.")
