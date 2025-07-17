# Handwritten Digit Classification

This is a simple Python application for **handwritten digit recognition** using a `RandomForestClassifier` from `scikit-learn`.  
It uses a `Tkinter` GUI that lets you **draw a digit**, **train the model**, and **predict** what digit you wrote.

---

## 📌 Features

- Draw digits (0–9) using your mouse.
- Train a simple Random Forest model on MNIST or scikit-learn’s Digits dataset.
- Save and load the trained model.
- Predict handwritten digits from your canvas input.
- Simple, clean GUI built with `Tkinter`.

---

## 🚀 Requirements

- Python 3.x
- `scikit-learn`
- `numpy`
- `Pillow`

Install dependencies:
```bash
pip install scikit-learn numpy pillow
```


## 📈 Project Motivation

The goal of this project is to show how classic machine learning algorithms can be used for image classification tasks and to demonstrate building a simple yet effective GUI for real-time handwritten digit recognition.

---

## 🎯 Learning Goals

- Build & train a Random Forest model on MNIST.
- Build an interactive GUI with Tkinter.
- Learn model serialization with pickle.
- Combine ML & UI in a single Python project.

---

## 💡 Possible Improvements

- Upgrade to deep learning (e.g., CNNs with TensorFlow/Keras).
- Add settings for hyperparameter tuning.
- Include real-time drawing feedback.
- Package the project as an executable for Windows/Linux.
- Add unit tests for model & GUI functions.

---
## 📊 Model Performance

| Metric                | Value (Approx.)               |
|-----------------------|--------------------------------|
| **Algorithm**         | Random Forest Classifier      |
| **Dataset**           | MNIST (28×28) or Digits (8×8) |
| **Training Accuracy** | ~95% (MNIST), ~98% (Digits)   |
| **Test Accuracy**     | ~90–95% (MNIST), ~95–99% (Digits) |
| **Precision**         | ~0.90–0.95 per class          |
| **Recall**            | ~0.90–0.95 per class          |
| **F1-Score**          | ~0.90–0.95 per class          |
| **Classes**           | Digits 0–9                    |
| **Confusion Matrix**  | Available via `classification_report` |

---

## 🧩 Dependencies

| Package | Version (Recommended) |
|---------|-----------------------|
| Python  | >= 3.8                |
| scikit-learn | >= 1.0          |
| numpy   | >= 1.21               |
| pillow  | >= 8.0                 |

---

## 👤 Author

**Monal Prashanth**  
[GitHub Profile](https://github.com/monal95)<br><br>
**Avantika**<br><br>
**Harsith**

