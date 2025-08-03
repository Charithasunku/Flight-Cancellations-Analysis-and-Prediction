from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model's performance on the test set.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy (tuned model): {accuracy:.4f}")
    print("Classification Report (tuned model):")
    print(report)
    print("Confusion Matrix (tuned model):")
    print(conf_matrix)

def cross_validate_model(model, X, y):
    """
    Performs cross-validation on the model.
    """
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"\nCross-validation Accuracy Scores: {cv_scores}")
    print(f"Mean Cross-validation Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
