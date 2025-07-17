# ✏️ Handwritten Digit Classification

This project is a **Python application for handwritten digit recognition**, with **two interfaces**:
1. A **Tkinter GUI**
2. An **interactive Streamlit web app**

Both use **classic machine learning** (Random Forest, SVM, MLP) with **advanced feature engineering** for robust handwritten digit classification.

---
## 🚀 Try it Live

Want to see the Streamlit version in action?  
👉 [**Launch Streamlit App**](https://YOUR-STREAMLIT-URL.streamlit.app)

Click the link above to draw digits, run predictions, and see the AI in your browser!

---
## 📌 Features

### 🖌️ Tkinter GUI
- Draw digits (0–9) using your mouse.
- Train a simple Random Forest model on MNIST or scikit-learn’s Digits dataset.
- Save/load the trained model.
- Predict handwritten digits from your canvas input.
- Simple, clean GUI built with `Tkinter`.

### 🌐 Streamlit Web App
- Draw digits directly in your browser using a canvas.
- Advanced preprocessing: histogram, projections, moments, zoning.
- Data augmentation: rotation, translation.
- Multiple model options: **RandomForest**, **SVM**, **MLP**.
- Optional hyperparameter tuning with Grid Search.
- Live prediction with confidence scores and probability charts.
- Confusion matrix and performance metrics.

---

## 🚀 Requirements

✅ **Common**
- Python 3.x

✅ **Tkinter Version**
- `scikit-learn`
- `numpy`
- `Pillow`
- `tkinter` (comes pre-installed with Python)

✅ **Streamlit Version**
- `streamlit`
- `streamlit-drawable-canvas`
- `scikit-learn`
- `numpy`
- `Pillow`
- `matplotlib`
- `seaborn`
- `opencv-python`

### 📦 Install dependencies

Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
## 📈 Project Motivation

The goal of this project is to demonstrate how **classic machine learning algorithms** can solve image classification tasks like handwritten digit recognition.  
It also shows how to deploy the same model using two approaches:
- A **desktop GUI** built with **Tkinter**
- A **modern interactive web app** built with **Streamlit**

By combining feature engineering, data augmentation, and user-friendly interfaces, this project helps learners understand how to bridge **ML models** and **real-world applications**.

## 🎯 Learning Goals

- Learn to build and train classic ML models (**Random Forest**, **SVM**, **MLP**) on image datasets like **MNIST** and **Digits**.
- Understand and apply **advanced feature engineering**: histograms, projections, moments, and zoning.
- Use **data augmentation** (rotations, translations) to improve generalization.
- Learn how to build an **interactive desktop GUI** using **Tkinter**.
- Deploy an **interactive ML web app** using **Streamlit**.
- Master **model serialization** using **pickle** to save and load trained models.
- Visualize model performance using confusion matrices and classification reports.

## 💡 Possible Improvements

- Upgrade to **deep learning** with Convolutional Neural Networks (**CNNs**) using TensorFlow or Keras for higher accuracy.
- Add more **hyperparameter controls** in both the Tkinter GUI and the Streamlit app.
- Package the **Tkinter app** as a standalone executable for Windows/Linux (using tools like PyInstaller).
- Deploy the **Streamlit app** on **Streamlit Community Cloud** or a public server.
- Add **unit tests** for model training, prediction, and preprocessing pipelines.
- Enhance the **drawing canvas** with brush size, color options, and real-time feedback.
- Add **multi-digit** or character recognition support.

## 🧩 How to Run

### ▶️ Run the Tkinter GUI

Make sure you have `tkinter_app.py` in your project folder, then run:
```bash
python tkinter_app.py
```
---

**📌 Note:** Adjust the filenames (`tkinter_app.py` and `streamlit_app.py`) to match your actual file names if they differ.


## 📊 Model Metrics & Accuracy

| Metric                | MNIST Dataset (28×28)         | Digits Dataset (8×8)         |
|-----------------------|--------------------------------|------------------------------|
| **Algorithm**         | Random Forest, SVM, MLP       | Random Forest, SVM, MLP      |
| **Training Accuracy** | ~95%–98%                      | ~98%–99%                     |
| **Test Accuracy**     | ~90%–95%                      | ~95%–99%                     |
| **Precision**         | ~0.90–0.95 per digit          | ~0.95–0.99 per digit         |
| **Recall**            | ~0.90–0.95 per digit          | ~0.95–0.99 per digit         |
| **F1-Score**          | ~0.90–0.95 per digit          | ~0.95–0.99 per digit         |
| **Classes**           | Digits 0–9                    | Digits 0–9                   |
| **Confusion Matrix**  | Available in `classification_report` & Streamlit UI | Available in `classification_report` & Streamlit UI |

---

## 📊 Example Model Performance

| Metric                | Value (Approx.)               |
|-----------------------|--------------------------------|
| **Algorithm**         | Random Forest, SVM, MLP       |
| **Dataset**           | MNIST (28×28) or Digits (8×8) |
| **Training Accuracy** | ~95% (MNIST), ~98% (Digits)   |
| **Test Accuracy**     | ~90–95% (MNIST), ~95–99% (Digits) |
| **Precision**         | ~0.90–0.95 per class          |
| **Recall**            | ~0.90–0.95 per class          |
| **F1-Score**          | ~0.90–0.95 per class          |
| **Classes**           | Digits 0–9                    |
| **Confusion Matrix**  | Available via `classification_report` and Streamlit UI |

---
## 👤 Authors

**Monal Prashanth**  
[GitHub](https://github.com/monal95)

**Avantika**

**Harsith**


## 📜 License

This project is licensed under the **MIT License**.
