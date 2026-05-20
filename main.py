import os
import pandas as pd

# FAST_RUN=1 (default): ~minutes. Use FAST_RUN=0 for heavier paper-style budgets.
FAST_RUN = os.environ.get("FAST_RUN", "1").strip().lower() not in ("0", "false", "no")
SAMPLE_PER_CLASS = 1200 if FAST_RUN else 4000

df = pd.read_csv('datasets/data.csv')
df = df[df['label'] != 1]
df['label'] = df['label'].apply(lambda x: 1 if x == 2 else 0)
df_0 = df[df['label'] == 0]
df_1 = df[df['label'] == 1]
n0 = min(SAMPLE_PER_CLASS, len(df_0))
n1 = min(SAMPLE_PER_CLASS, len(df_1))
df_0 = df_0.sample(n=n0, random_state=42)
df_1 = df_1.sample(n=n1, random_state=42)
df = pd.concat([df_0, df_1], ignore_index=True)

# =============================================================================
# Spam vs ham: ML pipeline with 95% CIs and p-values (outputs in English)
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import json
import joblib
from pathlib import Path
from scipy import stats
warnings.filterwarnings('ignore')

# All figures, tables, and model artifacts go to `result/`
try:
    _PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    _PROJECT_ROOT = Path.cwd()
RESULT_DIR = _PROJECT_ROOT / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split, permutation_test_score)
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
from xgboost import XGBClassifier
import shap

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120
RANDOM_STATE = 42


def _xgb_major_version():
    """Infer XGBoost major version (sklearn __init__ may hide `device` in **kwargs)."""
    import xgboost as xgb

    tag = xgb.__version__.split("+")[0].split("-")[0]
    head = "".join(c for c in tag if (c.isdigit() or c == "."))
    if not head:
        return 2
    lead = head.split(".")[0]
    return int(lead) if lead.isdigit() else 2


def _xgb_device_params():
    """Use GPU when CUDA build detected; tree_method.gpu_hist exists only on XGBoost 1.x."""
    import xgboost as xgb

    major = _xgb_major_version()
    try:
        bi = xgb.build_info()
    except Exception:
        bi = {}
    use_cuda_build = False
    if isinstance(bi, dict):
        flag = bi.get("USE_CUDA")
        use_cuda_build = flag is True or (
            isinstance(flag, str) and flag.upper() == "ON"
        )
    if not use_cuda_build:
        print(
            ">>> XGBoost: CPU (pip wheel built without CUDA). "
            "GPU: https://xgboost.readthedocs.io/en/stable/install.html"
        )
        if major >= 2:
            return {"tree_method": "hist", "device": "cpu", "n_jobs": -1}
        return {"tree_method": "hist", "n_jobs": -1}

    print(">>> XGBoost: GPU enabled.")
    if major >= 2:
        return {"tree_method": "hist", "device": "cuda", "n_jobs": 1}
    return {
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "n_jobs": 1,
    }


XGB_DEVICE_KW = _xgb_device_params()

if FAST_RUN:
    N_BOOTSTRAP = 120
    N_PERMUT_CV = 12
    N_PERMUT_HOLDOUT = 199
    CV_N_SPLITS = 5
    SHAP_MAX_ROWS = 900
    RF_N_EST = 120
    XGB_N_EST = 150
else:
    N_BOOTSTRAP = 300
    N_PERMUT_CV = 48
    N_PERMUT_HOLDOUT = 499
    CV_N_SPLITS = 8
    SHAP_MAX_ROWS = 2000
    RF_N_EST = 300
    XGB_N_EST = 400

if "N_JOBS_CV" in os.environ:
    N_JOBS_CV = int(os.environ["N_JOBS_CV"])
elif str(XGB_DEVICE_KW.get("device", "")).lower() == "cuda":
    N_JOBS_CV = 1
else:
    N_JOBS_CV = max(1, min((os.cpu_count() or 4), 8))

print(
    f"\n>>> Run profile: FAST_RUN={FAST_RUN} | per-class sample cap={SAMPLE_PER_CLASS} "
    f"| folds={CV_N_SPLITS} | CV permutations(F1)={N_PERMUT_CV} | bootstrap={N_BOOTSTRAP}"
)
print(f"    Parallel CV/permutation splits: n_jobs={N_JOBS_CV} (env N_JOBS_CV overrides; use 1 on GPU if OOM)")
print()

# =============================================================================
# 1. Load and inspect data
# =============================================================================
df = df.dropna(subset=['text', 'label']).reset_index(drop=True)
df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(int)

print("Data shape:", df.shape)
print("Class counts:")
print(df['label'].value_counts())

plt.figure(figsize=(5, 4))
sns.countplot(x='label', data=df, palette=['#4C72B0', '#DD8452'])
plt.xticks([0, 1], ['Ham (0)', 'Spam (1)'])
plt.title("Figure 1. Class distribution")
plt.savefig(RESULT_DIR / "fig1_class_distribution.png", bbox_inches='tight')
plt.show()

# =============================================================================
# 2. Hold-out test set (30%) — never used for training / CV / tuning
# =============================================================================
df_train, df_holdout = train_test_split(
    df,
    test_size=0.30,
    stratify=df['label'],
    random_state=RANDOM_STATE
)
df_train   = df_train.reset_index(drop=True)
df_holdout = df_holdout.reset_index(drop=True)

print(f"\n>>> Train / hold-out split:")
print(f"Train (70%):     {df_train.shape[0]} rows")
print(f"Hold-out (30%):  {df_holdout.shape[0]} rows (locked until final eval)")

# =============================================================================
# 3. Text preprocessing (stop-word removal)
# =============================================================================
STOP_WORDS = set(ENGLISH_STOP_WORDS)
KEEP_TOKENS = {"url", "num"}
STOP_WORDS = STOP_WORDS - KEEP_TOKENS


def clean_text(t):
    """Lowercase, URL/number placeholders, strip non-letters, remove stop words."""
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " URL ", t)
    t = re.sub(r"\b\d+\b", " NUM ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    tokens = [w for w in t.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)

df_train['clean_text']   = df_train['text'].apply(clean_text)
df_holdout['clean_text'] = df_holdout['text'].apply(clean_text)

# =============================================================================
# 4. Features: TF-IDF (+ export processed artifacts)
# =============================================================================
tfidf = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=10000,
    min_df=5,
    sublinear_tf=True
)

X_train_full = tfidf.fit_transform(df_train['clean_text'])
y_train_full = df_train['label'].values

X_holdout = tfidf.transform(df_holdout['clean_text'])
y_holdout = df_holdout['label'].values

feature_names = np.array(tfidf.get_feature_names_out())
print(f"\nTrain matrix:    {X_train_full.shape}")
print(f"Hold-out matrix: {X_holdout.shape}")

print("\n>>> Saving TF-IDF artifacts ...")

from scipy.sparse import save_npz, vstack
X_combined = vstack([X_train_full, X_holdout])
save_npz(RESULT_DIR / "processed_tfidf_sparse.npz", X_combined)

train_dense   = pd.DataFrame(X_train_full.toarray(), columns=feature_names)
train_dense['label']   = y_train_full
train_dense['dataset'] = 'train'
train_dense['clean_text'] = df_train['clean_text'].values

holdout_dense = pd.DataFrame(X_holdout.toarray(), columns=feature_names)
holdout_dense['label']   = y_holdout
holdout_dense['dataset'] = 'holdout'
holdout_dense['clean_text'] = df_holdout['clean_text'].values

processed_df = pd.concat([train_dense, holdout_dense], axis=0).reset_index(drop=True)
processed_df.to_csv(RESULT_DIR / "processed_tfidf.csv", index=False)
print(f"Saved: {RESULT_DIR / 'processed_tfidf.csv'} (shape={processed_df.shape})")
print(f"Saved: {RESULT_DIR / 'processed_tfidf_sparse.npz'} (sparse)")

joblib.dump(tfidf, RESULT_DIR / "tfidf_vectorizer.pkl")
print(f"Saved: {RESULT_DIR / 'tfidf_vectorizer.pkl'}")

# =============================================================================
# 5. Models
# =============================================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0,
                                              solver='liblinear',
                                              random_state=RANDOM_STATE),
    "Random Forest"     : RandomForestClassifier(n_estimators=RF_N_EST,
                                                 n_jobs=1,
                                                 random_state=RANDOM_STATE),
    "XGBoost"           : XGBClassifier(n_estimators=XGB_N_EST, max_depth=6,
                                        learning_rate=0.1,
                                        use_label_encoder=False,
                                        eval_metric='logloss',
                                        random_state=RANDOM_STATE,
                                        **XGB_DEVICE_KW),
}

# =============================================================================
# 6. Statistics helpers: 95% CI + p-values
# =============================================================================
def ci_from_folds(scores, alpha=0.05):
    """95% CI for CV fold scores via t distribution on fold means."""
    n = len(scores)
    mean = np.mean(scores)
    se   = stats.sem(scores)
    h    = se * stats.t.ppf(1 - alpha/2, n - 1)
    return mean, mean - h, mean + h

def bootstrap_metric(y_true, y_pred_or_proba, metric_fn, n_boot=300, alpha=0.05,
                     seed=RANDOM_STATE):
    """Bootstrap 95% CI for a metric on hold-out predictions/probabilities."""
    rng    = np.random.RandomState(seed)
    n      = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            s = metric_fn(y_true[idx], y_pred_or_proba[idx])
            scores.append(s)
        except Exception:
            continue
    scores = np.array(scores)
    point  = metric_fn(y_true, y_pred_or_proba)
    lo, hi = np.percentile(scores, [100*alpha/2, 100*(1-alpha/2)])
    return point, lo, hi, scores

def permutation_pvalue(y_true, y_pred_or_proba, metric_fn, n_perm=499,
                       seed=RANDOM_STATE):
    """
    Shuffle true labels vs fixed predictions/probabilities (cheap null).
    One-sided p-value: P(perm_metric >= observed).
    """
    rng        = np.random.RandomState(seed)
    observed   = metric_fn(y_true, y_pred_or_proba)
    perm_scores = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y_true)
        try:
            perm_scores.append(metric_fn(y_perm, y_pred_or_proba))
        except Exception:
            continue
    perm_scores = np.array(perm_scores)
    p_value = (np.sum(perm_scores >= observed) + 1) / (len(perm_scores) + 1)
    return observed, p_value

# =============================================================================
# 7. Stratified CV (+ 95% CI on folds + permutation p-value for F1)
# =============================================================================
scoring = {'accuracy': 'accuracy', 'precision': 'precision',
           'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}
cv = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

cv_results = []
cv_raw_scores = {}
permutation_pvals = {}

for name, model in models.items():
    print(f"\n>>> Cross-validating: {name} ...")
    res = cross_validate(model, X_train_full, y_train_full,
                         cv=cv, scoring=scoring,
                         n_jobs=N_JOBS_CV, return_train_score=False)
    cv_raw_scores[name] = res

    row = {"Model": name}
    for m in ["accuracy","precision","recall","f1","roc_auc"]:
        scores = res[f'test_{m}']
        mean, lo, hi = ci_from_folds(scores)
        row[m.upper()] = f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"
        row[f"_{m}_mean"] = mean
        row[f"_{m}_std"]  = scores.std()

    print(f"    Permutation test (F1), n_perm={N_PERMUT_CV} ...")
    _, _, p_val = permutation_test_score(
        model, X_train_full, y_train_full,
        scoring='f1', cv=cv, n_permutations=N_PERMUT_CV,
        n_jobs=N_JOBS_CV, random_state=RANDOM_STATE
    )
    row["p-value (F1)"] = f"{p_val:.4g}"
    permutation_pvals[name] = p_val

    cv_results.append(row)

cv_df = pd.DataFrame(cv_results).sort_values("_f1_mean", ascending=False)
display_cols = ["Model","ACCURACY","PRECISION","RECALL","F1","ROC_AUC","p-value (F1)"]
cv_table = cv_df[display_cols].copy()
print(f"\n=== Table 1. Stratified {CV_N_SPLITS}-fold CV — mean [95% CI] ===")
print(cv_table.to_string(index=False))
cv_table.to_csv(RESULT_DIR / "table1_cv_results.csv", index=False)

print("\n=== Table 1b. Pairwise paired t-tests on fold F1 ===")
pair_pvals = []
model_names = list(models.keys())
for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        a, b = model_names[i], model_names[j]
        f1_a = cv_raw_scores[a]['test_f1']
        f1_b = cv_raw_scores[b]['test_f1']
        t_stat, p_val = stats.ttest_rel(f1_a, f1_b)
        pair_pvals.append({
            "Model A": a, "Model B": b,
            "Mean Diff (F1)": f"{(f1_a.mean()-f1_b.mean()):+.4f}",
            "t-statistic"   : f"{t_stat:.3f}",
            "p-value"       : f"{p_val:.4g}",
            "Significant (alpha=0.05)": "Yes" if p_val < 0.05 else "No"
        })
pair_df = pd.DataFrame(pair_pvals)
print(pair_df.to_string(index=False))
pair_df.to_csv(RESULT_DIR / "table1b_pairwise_ttest.csv", index=False)

fold_long = []
for name, res in cv_raw_scores.items():
    for m in ["accuracy","precision","recall","f1","roc_auc"]:
        for s in res[f'test_{m}']:
            fold_long.append({"Model":name, "Metric":m.upper(), "Score":s})
fold_long_df = pd.DataFrame(fold_long)

plt.figure(figsize=(11,5))
sns.boxplot(data=fold_long_df, x="Metric", y="Score", hue="Model")
plt.title(f"Figure 2. CV score distribution ({CV_N_SPLITS}-fold stratified)")
ymin = float(fold_long_df["Score"].min())
plt.ylim(max(0.0, ymin - 0.05), min(1.02, fold_long_df["Score"].max() + 0.03))
plt.legend(loc="lower right")
plt.savefig(RESULT_DIR / "fig2_model_comparison.png", bbox_inches='tight')
plt.show()

# =============================================================================
# 8. Refit best model (by mean CV F1) on full train split
# =============================================================================
best_name  = cv_df.iloc[0]["Model"]
print(f"\n*** Best model by mean CV F1: {best_name} ***")
best_model = models[best_name].fit(X_train_full, y_train_full)

# =============================================================================
# 9. Final evaluation on 30% hold-out (Bootstrap CI + label-shuffle p)
# =============================================================================
print("\n" + "="*72)
print(">>> Final evaluation — 30% hold-out (never used for model selection)")
print("="*72)

y_pred  = best_model.predict(X_holdout)
y_proba = best_model.predict_proba(X_holdout)[:,1]

print(f"\n=== Classification report (hold-out | {best_name}) ===")
print(classification_report(y_holdout, y_pred, target_names=['Ham', 'Spam']))

_prf_kw = {'zero_division': 0}
metric_fns = {
    "Accuracy": (accuracy_score, y_pred),
    "Precision": (partial(precision_score, **_prf_kw), y_pred),
    "Recall": (partial(recall_score, **_prf_kw), y_pred),
    "F1-Score": (partial(f1_score, **_prf_kw), y_pred),
    "ROC-AUC": (roc_auc_score, y_proba),
}

holdout_rows = []
print("\n>>> Bootstrap CIs & label-shuffle p-values on hold-out ...")
for metric_name, (fn, preds) in metric_fns.items():
    point, lo, hi, _ = bootstrap_metric(y_holdout, preds, fn,
                                        n_boot=N_BOOTSTRAP)
    _, p_val = permutation_pvalue(y_holdout, preds, fn, n_perm=N_PERMUT_HOLDOUT)
    holdout_rows.append({
        "Metric"    : metric_name,
        "Point Est.": f"{point:.4f}",
        "95% CI"    : f"[{lo:.4f}, {hi:.4f}]",
        "p-value"   : f"{p_val:.4g}",
        "_point"    : point,
        "_lo"       : lo,
        "_hi"       : hi,
        "_p"        : p_val
    })

holdout_df = pd.DataFrame(holdout_rows)
print("\n=== Table 6. Hold-out performance (Bootstrap 95% CI + permutation p-value) ===")
print(holdout_df[["Metric","Point Est.","95% CI","p-value"]].to_string(index=False))
holdout_df[["Metric","Point Est.","95% CI","p-value"]].to_csv(
    RESULT_DIR / "table6_holdout_results.csv", index=False)

plt.figure(figsize=(8, 5))
xs = np.arange(len(holdout_df))
plt.errorbar(xs, holdout_df["_point"],
             yerr=[holdout_df["_point"]-holdout_df["_lo"],
                   holdout_df["_hi"]-holdout_df["_point"]],
             fmt='o', capsize=8, markersize=10, color='#4C72B0',
             linewidth=2)
plt.xticks(xs, holdout_df["Metric"])
plt.ylim(max(0.0, holdout_df["_point"].min() - 0.08), 1.02)
plt.ylabel("Score")
plt.title(f"Figure 3. Hold-out scores with Bootstrap 95% CI ({best_name})")
plt.grid(alpha=0.3)
plt.savefig(RESULT_DIR / "fig3_holdout_ci.png", bbox_inches='tight')
plt.show()

cm = confusion_matrix(y_holdout, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
plt.title(f"Figure 4. Confusion matrix (hold-out) — {best_name}")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.savefig(RESULT_DIR / "fig4_confusion_matrix.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(7, 6))
for name, mdl in models.items():
    mdl.fit(X_train_full, y_train_full)
    proba = mdl.predict_proba(X_holdout)[:,1]
    fpr, tpr, _ = roc_curve(y_holdout, proba)
    auc_val, auc_lo, auc_hi, _ = bootstrap_metric(
        y_holdout, proba, roc_auc_score, n_boot=N_BOOTSTRAP)
    plt.plot(fpr, tpr,
             label=f"{name} (AUC={auc_val:.3f}, 95% CI [{auc_lo:.3f}, {auc_hi:.3f}])")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Figure 5. ROC curves — hold-out (Bootstrap AUC 95% CI in legend)")
plt.legend(loc="lower right", fontsize=9)
plt.savefig(RESULT_DIR / "fig5_roc_curves.png", bbox_inches='tight')
plt.show()

# =============================================================================
# 10. Feature importance (coefficients / tree gain)
# =============================================================================
lr = LogisticRegression(max_iter=2000, C=1.0,
                        solver='liblinear',
                        random_state=RANDOM_STATE).fit(X_train_full, y_train_full)
coefs = lr.coef_[0]
odds_ratio = np.exp(coefs)

lr_imp = pd.DataFrame({
    "Feature"    : feature_names,
    "Coefficient": coefs,
    "Odds_Ratio" : odds_ratio
})
top_spam = lr_imp.sort_values("Coefficient", ascending=False).head(20)
top_ham  = lr_imp.sort_values("Coefficient", ascending=True ).head(20)

print("\n=== Table 2. Top 20 tokens aligned with spam (logistic coefficient / OR)")
print(top_spam.to_string(index=False))
top_spam.to_csv(RESULT_DIR / "table2_top_spam_features.csv", index=False)

print("\n=== Table 3. Top 20 tokens aligned with ham (coefficient / OR)")
print(top_ham.to_string(index=False))
top_ham.to_csv(RESULT_DIR / "table3_top_ham_features.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 7))
sns.barplot(y="Feature", x="Odds_Ratio", data=top_spam,
            palette="Reds_r", ax=axes[0])
axes[0].set_title("Top 20 spam-like tokens (odds ratio > 1)")
axes[0].axvline(1, color='black', linestyle='--')

sns.barplot(y="Feature", x="Odds_Ratio", data=top_ham,
            palette="Blues_r", ax=axes[1])
axes[1].set_title("Top 20 ham-like tokens (odds ratio < 1)")
axes[1].axvline(1, color='black', linestyle='--')
fig.suptitle("Figure 6. Logistic regression — odds ratios for top discriminative tokens", y=1.02)
plt.tight_layout()
plt.savefig(RESULT_DIR / "fig6_odds_ratio.png", bbox_inches='tight')
plt.show()
plt.close(fig)

if best_name in ["Random Forest", "XGBoost"]:
    importances = best_model.feature_importances_
    imp_df = pd.DataFrame({"Feature":feature_names,
                           "Importance":importances})
    top_imp = imp_df.sort_values("Importance", ascending=False).head(25)

    plt.figure(figsize=(8,8))
    sns.barplot(y="Feature", x="Importance", data=top_imp, palette="viridis")
    plt.title(f"Figure 7. Top 25 features by tree importance ({best_name})")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "fig7_feature_importance.png", bbox_inches='tight')
    plt.show()
    top_imp.to_csv(RESULT_DIR / "table4_feature_importance.csv", index=False)

# =============================================================================
# 11. SHAP — positive class = spam (class index 1)
# =============================================================================
def _shap_positive_class(arr):
    """Normalize TreeExplainer / LinearExplainer output to (n_samples, n_features) for spam."""
    if isinstance(arr, list):
        return np.asarray(arr[1])
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] >= 2:
        return a[..., 1]
    return a


sample_idx = np.random.RandomState(RANDOM_STATE).choice(
    X_holdout.shape[0], size=min(SHAP_MAX_ROWS, X_holdout.shape[0]), replace=False
)
X_sample = X_holdout[sample_idx]
X_sample_dense = X_sample.toarray()

if best_name == "XGBoost":
    explainer = shap.TreeExplainer(best_model)
    shap_vals = _shap_positive_class(explainer.shap_values(X_sample))
elif best_name == "Random Forest":
    explainer = shap.TreeExplainer(best_model)
    shap_vals = _shap_positive_class(explainer.shap_values(X_sample))
else:
    explainer = shap.LinearExplainer(best_model, X_train_full,
                                     feature_perturbation="interventional")
    shap_vals = _shap_positive_class(explainer.shap_values(X_sample))

plt.figure(figsize=(11, 7))
shap.summary_plot(
    shap_vals,
    X_sample_dense,
    feature_names=feature_names,
    plot_type="dot",
    max_display=20,
    show=False,
    title="Figure 8. SHAP — contribution toward spam (positive class)",
)
fig_shap = plt.gcf()
plt.tight_layout()
plt.savefig(RESULT_DIR / "fig8_shap_summary.png", bbox_inches="tight", dpi=150)
plt.show()
plt.close(fig_shap)

mean_abs = np.abs(shap_vals).mean(axis=0)
shap_rank = pd.DataFrame({"Feature": feature_names,
                          "Mean_Abs_SHAP": mean_abs}) \
              .sort_values("Mean_Abs_SHAP", ascending=False).head(25)
print("\n=== Table 5. Top 25 tokens by mean |SHAP| (spam class) ===")
print(shap_rank.to_string(index=False))
shap_rank.to_csv(RESULT_DIR / "table5_shap_importance.csv", index=False)

# =============================================================================
# 12. Save summary + bundles
# =============================================================================
final_summary = {
    "Best Model"     : best_name,
    "Evaluation Set" : "30% Hold-out Validation Set",
    "Holdout_Size"   : len(y_holdout),
}
for row in holdout_rows:
    final_summary[f"{row['Metric']} (point)"]  = row["_point"]
    final_summary[f"{row['Metric']} 95% CI"]   = f"[{row['_lo']:.4f}, {row['_hi']:.4f}]"
    final_summary[f"{row['Metric']} p-value"]  = row["_p"]

pd.DataFrame([final_summary]).to_csv(RESULT_DIR / "final_summary.csv", index=False)

print("\n=== Final summary (hold-out validation set) ===")
for k, v in final_summary.items():
    print(f"{k}: {v}")

print("\n>>> Saving model bundle ...")
model_bundle = {
    "model"          : best_model,
    "vectorizer"     : tfidf,
    "model_name"     : best_name,
    "feature_names"  : feature_names,
    "stop_words"     : list(STOP_WORDS),
    "random_state"   : RANDOM_STATE,
    "training_size"  : X_train_full.shape[0],
    "holdout_metrics": {row["Metric"]: {
                          "point" : row["_point"],
                          "ci_low": row["_lo"],
                          "ci_high": row["_hi"],
                          "p_value": row["_p"]
                       } for row in holdout_rows}
}
joblib.dump(model_bundle, RESULT_DIR / "best_model.pkl")
joblib.dump(best_model, RESULT_DIR / "best_model_only.pkl")
print(f"Saved: {RESULT_DIR / 'best_model.pkl'} (model + vectorizer + metadata)")
print(f"Saved: {RESULT_DIR / 'best_model_only.pkl'} (model only)")

# --- Browser demo: TF-IDF + logistic weights (pure HTML/JS frontend) -----
WEB_DIR = _PROJECT_ROOT / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)
browser_surrogate_lr = False
if isinstance(best_model, LogisticRegression):
    lr_browser = best_model
else:
    browser_surrogate_lr = True
    lr_browser = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear",
                                    random_state=RANDOM_STATE)
    lr_browser.fit(X_train_full, y_train_full)
idf_browser = getattr(tfidf, "idf_", None)
if idf_browser is None:
    idf_browser = np.ones(len(feature_names))
browser_pack = {
    "version": 1,
    "best_model_cv": best_name,
    "inference_mode": ("surrogate_logistic"
                       if browser_surrogate_lr else "best_logistic_regression"),
    "note": ("Probabilities match the CV-winning model only when it is "
             "sklearn LogisticRegression; otherwise this uses an LR surrogate "
             "fit on identical TF-IDF features for consistent browser inference."),
    "ngram_range": list(tfidf.ngram_range),
    "sublinear_tf": bool(tfidf.sublinear_tf),
    "smooth_idf": bool(getattr(tfidf, "smooth_idf", True)),
    "norm": (getattr(tfidf, "norm", None) or "l2"),
    "intercept": float(np.asarray(lr_browser.intercept_).ravel()[0]),
    "coef": np.asarray(lr_browser.coef_).ravel().astype(np.float64).tolist(),
    "feature_names": feature_names.tolist(),
    "idf": np.asarray(idf_browser).ravel().astype(np.float64).tolist(),
    "stop_words": sorted(STOP_WORDS),
    "keep_tokens": sorted(KEEP_TOKENS),
}
out_web = WEB_DIR / "model.json"
with open(out_web, "w", encoding="utf-8") as fh:
    json.dump(browser_pack, fh, separators=(",", ":"))
with open(RESULT_DIR / "browser_model.json", "w", encoding="utf-8") as fh:
    json.dump(browser_pack, fh, separators=(",", ":"))
print(f"Saved: {out_web} (TF-IDF + logistic for spam_checker_demo)")

print("\n>>> Example: load bundle and run inference")
print("-" * 60)
print("""
import joblib

bundle = joblib.load("result/best_model.pkl")
model = bundle["model"]
vectorizer = bundle["vectorizer"]

new_texts = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts!",
    "Hey, are we still meeting tomorrow?",
]
new_clean = [clean_text(t) for t in new_texts]
X_new = vectorizer.transform(new_clean)
preds = model.predict(X_new)
probas = model.predict_proba(X_new)[:, 1]
for txt, p, pr in zip(new_texts, preds, probas):
    tag = "SPAM" if p == 1 else "HAM"
    print(f"[{tag}] (p_spam={pr:.3f}) {txt[:60]}")
""")

print("\n" + "="*72)
print(f"[All outputs written under: {RESULT_DIR.resolve()}]")
print("="*72)
print("""
Artifacts (result/):
  Figures: fig1–fig8 (.png)
  Tables: table1–table6 (.csv), final_summary.csv
  Processed: processed_tfidf.csv, processed_tfidf_sparse.npz
  Models: best_model.pkl, best_model_only.pkl, tfidf_vectorizer.pkl
<<<<<<< HEAD
  Web demo: web/model.json + web/index.html (serve web/ over HTTP to load JSON)
=======
  Web demo: web/model.json + open web/index.html via a local server
>>>>>>> c4ade16f2ff66f8c6121d14d77025876449c0ab8

Speed: default FAST_RUN=1. Heavier run: FAST_RUN=0 python main.py
""")