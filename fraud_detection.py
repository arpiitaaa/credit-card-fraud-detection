import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
)

df = pd.read_csv("creditcard.csv")
df[["Amount", "Time"]] = StandardScaler().fit_transform(df[["Amount", "Time"]])

X, y = df.drop("Class", axis=1), df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ratio of legit to fraud, used by XGBoost to handle imbalance
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced_subsample",
        min_samples_leaf=2, n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos, eval_metric="aucpr",
        n_jobs=-1, random_state=42
    ),
}

os.makedirs("plots", exist_ok=True)
roc_fig, roc_ax = plt.subplots(figsize=(6, 5))
pr_fig, pr_ax = plt.subplots(figsize=(6, 5))

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    print(f"\n=== {name} ===")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, digits=4))
    print("ROC-AUC:", round(auc, 4))
    print("Average Precision (PR-AUC):", round(ap, 4))

    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    precision, recall, _ = precision_recall_curve(y_test, proba)
    pr_ax.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")

roc_ax.plot([0, 1], [0, 1], "k--", label="Random")
roc_ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
roc_ax.legend(loc="lower right")
roc_fig.tight_layout()
roc_fig.savefig("plots/roc_curve.png", dpi=150)

pr_ax.axhline(y_test.mean(), color="k", linestyle="--", label="Random")
pr_ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
pr_ax.legend(loc="lower left")
pr_fig.tight_layout()
pr_fig.savefig("plots/precision_recall_curve.png", dpi=150)

plt.show()