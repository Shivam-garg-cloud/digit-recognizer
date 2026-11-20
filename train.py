"""Handwritten Digit Recognizer - compares three classic machine-learning models.

Uses the small digits dataset that ships with scikit-learn (1,797 images, 8x8 pixels,
labels 0-9), so no download, account or API key is needed.

Run:  python train.py
Creates in the outputs/ folder:
  results.csv          accuracy and training time of each model
  accuracy_chart.png   bar chart of the accuracies
  confusion_matrix.png confusion matrix of the best model
  sample_predictions.png  a few test images with the best model's predictions
  best_model.joblib    the saved best model
"""
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

OUT = Path(__file__).with_name("outputs")
SEED = 42


def load_data():
    digits = load_digits()
    return train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=SEED, stratify=digits.target
    ), digits


def get_models():
    return {
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    }


def main():
    OUT.mkdir(exist_ok=True)
    (X_train, X_test, y_train, y_test), digits = load_data()
    print(f"Training images: {len(X_train)}, test images: {len(X_test)}")

    rows, fitted = [], {}
    for name, model in get_models().items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start
        acc = accuracy_score(y_test, model.predict(X_test))
        fitted[name] = model
        rows.append({"model": name, "accuracy_percent": round(100 * acc, 2),
                     "train_time_seconds": round(train_time, 4)})
        print(f"{name}: accuracy {100 * acc:.2f}%  (trained in {train_time:.3f}s)")

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "results.csv", index=False)

    best_name = results.loc[results["accuracy_percent"].idxmax(), "model"]
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")
    joblib.dump(best, OUT / "best_model.joblib")

    # Chart 1: accuracy comparison
    plt.figure(figsize=(6, 4))
    plt.bar(results["model"], results["accuracy_percent"], color="#444444")
    plt.ylim(0, 105)
    plt.ylabel("Test accuracy (%)")
    plt.title("Model comparison on the digits test set")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUT / "accuracy_chart.png", dpi=150)
    plt.close()

    # Chart 2: confusion matrix of best model
    ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, cmap="Greys")
    plt.title(f"Confusion matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close()

    # Chart 3: sample predictions
    predictions = best.predict(X_test[:10])
    fig, axes = plt.subplots(2, 5, figsize=(8, 4))
    for ax, image, pred, true in zip(axes.ravel(), X_test[:10], predictions, y_test[:10]):
        ax.imshow(image.reshape(8, 8), cmap="gray_r")
        ax.set_title(f"pred {pred} / true {true}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "sample_predictions.png", dpi=150)
    plt.close()

    print(f"Files saved in {OUT}")


if __name__ == "__main__":
    main()
