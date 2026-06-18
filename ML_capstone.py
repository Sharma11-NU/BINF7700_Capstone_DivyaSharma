import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay,
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#  LOAD DATA

X_train        = pd.read_csv("X_train.csv")
X_test         = pd.read_csv("X_test.csv")
X_train_scaled = pd.read_csv("X_train_scaled.csv")
X_test_scaled  = pd.read_csv("X_test_scaled.csv")
y_train        = pd.read_csv("y_train.csv").squeeze()
y_test         = pd.read_csv("y_test.csv").squeeze()

print(f"✓ Data loaded | Train: {X_train.shape} | Test: {X_test.shape}")

# TRAIN MODELS

# Logistic Regression
lr = LogisticRegression(class_weight="balanced",
                        max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
lr_probs = lr.predict_proba(X_test_scaled)[:, 1]

# XGBoost
scale = (y_train == 0).sum() / (y_train == 1).sum()
xgb   = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       scale_pos_weight=scale, eval_metric="logloss",
                       random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_probs = xgb.predict_proba(X_test)[:, 1]

# Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=6,
                             class_weight="balanced",
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

print("✓ All three models trained")

# CROSS VALIDATION 

print("\n--- 5-Fold Cross Validation ---")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_cv  = cross_val_score(lr,  X_train_scaled, y_train, cv=cv, scoring="roc_auc")
xgb_cv = cross_val_score(xgb, X_train,        y_train, cv=cv, scoring="roc_auc")
rf_cv  = cross_val_score(rf,  X_train,        y_train, cv=cv, scoring="roc_auc")

print(f"  LR  CV AUC: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
print(f"  XGB CV AUC: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")
print(f"  RF  CV AUC: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

# METRICS TABLE

print("\n--- Full Metrics: Logistic Regression ---")
print(classification_report(y_test, lr_preds,
                             target_names=["No Toxicity", "Toxicity"]))
print(f"AUC-ROC  : {roc_auc_score(y_test, lr_probs):.4f}")
print(f"Accuracy : {(lr_preds == y_test).mean():.4f}")

print("\n--- Full Metrics: XGBoost ---")
print(classification_report(y_test, xgb_preds,
                             target_names=["No Toxicity", "Toxicity"]))
print(f"AUC-ROC  : {roc_auc_score(y_test, xgb_probs):.4f}")
print(f"Accuracy : {(xgb_preds == y_test).mean():.4f}")

print("\n--- Full Metrics: Random Forest ---")
print(classification_report(y_test, rf_preds,
                             target_names=["No Toxicity", "Toxicity"]))
print(f"AUC-ROC  : {roc_auc_score(y_test, rf_probs):.4f}")
print(f"Accuracy : {(rf_preds == y_test).mean():.4f}")

# CONFUSION MATRICES

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, preds, title in zip(
    axes,
    [lr_preds, xgb_preds, rf_preds],
    ["Logistic Regression", "XGBoost", "Random Forest"]
):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Toxicity", "Toxicity"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)

plt.suptitle("Confusion Matrices", fontsize=13)
plt.tight_layout()
plt.savefig("eval_confusion_matrices.png")
plt.show()

# ROC CURVES

fig, ax = plt.subplots(figsize=(7, 5))

for probs, label, color in zip(
    [lr_probs, xgb_probs, rf_probs],
    ["Logistic Regression", "XGBoost", "Random Forest"],
    ["steelblue", "darkorange", "seagreen"]
):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", color=color)

ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend()
plt.tight_layout()
plt.savefig("eval_roc_curves.png")
plt.show()

fig, ax = plt.subplots(figsize=(7, 5))

X_full        = pd.concat([X_train, X_test], ignore_index=True)
X_full_scaled = pd.concat([X_train_scaled, X_test_scaled], ignore_index=True)
y_full        = pd.concat([y_train, y_test], ignore_index=True)

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model, X_cv, label, color in zip(
    [lr,  xgb,   rf],
    [X_full_scaled, X_full, X_full],
    ["Logistic Regression", "XGBoost", "Random Forest"],
    ["blue", "orange", "green"]
):
    mean_recall = np.linspace(0, 1, 100)
    precisions  = []
    ap_scores   = []

    for train_idx, val_idx in cv5.split(X_cv, y_full):
        X_tr, X_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]

        prec, rec, _ = precision_recall_curve(y_val, probs)
        prec_interp  = np.interp(mean_recall, rec[::-1], prec[::-1])
        precisions.append(prec_interp)
        ap_scores.append(average_precision_score(y_val, probs))

    mean_prec = np.mean(precisions, axis=0)
    std_prec  = np.std(precisions, axis=0)
    mean_ap   = np.mean(ap_scores)

    ax.plot(mean_recall, mean_prec,
            label=f"{label} (mean AP = {mean_ap:.3f})", color=color)
    ax.fill_between(mean_recall,
                    mean_prec - std_prec,
                    mean_prec + std_prec,
                    alpha=0.15, color=color)

ax.axhline(y_full.mean(), color="k", linestyle="--",
           label=f"Random (AP = {y_full.mean():.3f})")
ax.set_ylim([0, 1.05])
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Cross-Validated Precision-Recall Curve (5-Fold)")
ax.legend()
plt.tight_layout()
plt.savefig("eval_precision_recall_cv.png")
plt.show()

# FEATURE IMPORTANCE 

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, model, title in zip(
    axes,
    [xgb, rf],
    ["XGBoost Feature Importance", "Random Forest Feature Importance"]
):
    feat_imp = pd.Series(model.feature_importances_,
                         index=X_train.columns).sort_values(ascending=True)
    feat_imp.plot(kind="barh", color="steelblue", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("eval_feature_importance.png")
plt.show()

# LOGISTIC REGRESSION COEFFICIENTS

coef = pd.Series(lr.coef_[0], index=X_train.columns).sort_values()

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in coef.values]
coef.plot(kind="barh", color=colors, ax=ax)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Logistic Regression Coefficients\n"
             "(Red = increases toxicity risk, Green = decreases)")
ax.set_xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("eval_lr_coefficients.png")
plt.show()

# SUMMARY

lr_auc  = roc_auc_score(y_test, lr_probs)
xgb_auc = roc_auc_score(y_test, xgb_probs)
rf_auc  = roc_auc_score(y_test, rf_probs)

lr_ap  = average_precision_score(y_test, lr_probs)
xgb_ap = average_precision_score(y_test, xgb_probs)
rf_ap  = average_precision_score(y_test, rf_probs)

print(f"""
AUC-ROC       : {lr_auc:.4f}       {xgb_auc:.4f}      {rf_auc:.4f}
Avg Precision : {lr_ap:.4f}       {xgb_ap:.4f}        {rf_ap:.4f}       
CV AUC        : {lr_cv.mean():.4f}±{lr_cv.std():.3f}  {xgb_cv.mean():.4f}±{xgb_cv.std():.3f}  {rf_cv.mean():.4f}±{rf_cv.std():.3f} 
Accuracy      : {(lr_preds==y_test).mean():.4f}       {(xgb_preds==y_test).mean():.4f}         {(rf_preds==y_test).mean():.4f} 
Dataset size  : 452 patients 
Limitation    : Small dataset, modest AUC                   

Plots saved:
  eval_confusion_matrices.png
  eval_roc_curves.png
  eval_precision_recall_cv.png
  eval_feature_importance.png
  eval_lr_coefficients.png
""")
