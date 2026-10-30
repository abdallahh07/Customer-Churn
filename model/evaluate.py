from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("Accuracy:", acc)
    print("ROC-AUC:", auc)
    print(classification_report(y_test, y_pred))

    return {"accuracy": acc, "roc_auc": auc}