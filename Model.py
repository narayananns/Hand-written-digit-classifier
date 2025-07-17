import os
import tkinter as tk
from tkinter import Canvas, Button, Label, Frame, messagebox
import numpy as np
from PIL import Image, ImageDraw
import pickle
import threading
import time
import sys

dataset_download_in_progress = False

class DigitClassifier:
    def __init__(self):
        self.model = None
        self.trained = False
        self.data_loaded = False

    def build_model(self):
        try:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
            return self.model
        except ImportError:
            print("Error: scikit-learn is not installed.")
            return None

    def load_data(self, status_callback=None):
        global dataset_download_in_progress

        if dataset_download_in_progress:
            if status_callback:
                status_callback("Dataset already downloading...")
            return None

        dataset_download_in_progress = True

        try:
            if status_callback:
                status_callback("Loading MNIST dataset...")

            try:
                from sklearn.datasets import fetch_openml
                mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
                X = mnist.data.astype('float32') / 255.0
                y = mnist.target.astype('int')
            except Exception:
                if status_callback:
                    status_callback("Failed to load MNIST. Loading smaller digits dataset...")
                from sklearn.datasets import load_digits
                digits = load_digits()
                X = digits.data / 16.0
                y = digits.target

            from sklearn.model_selection import train_test_split
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.data_loaded = True
            dataset_download_in_progress = False

            if status_callback:
                status_callback("Dataset loaded successfully!")
            return (self.x_train, self.y_train), (self.x_test, self.y_test)

        except Exception as e:
            print(f"Error loading data: {e}")
            if status_callback:
                status_callback(f"Error loading data: {e}")
            dataset_download_in_progress = False
            return None

    def train_model(self, status_callback=None):
        if self.model is None:
            self.build_model()

        if not self.data_loaded:
            if self.load_data(status_callback) is None:
                return None

        try:
            if status_callback:
                status_callback("Training model...")
            self.model.fit(self.x_train, self.y_train)
            self.trained = True

            score = self.model.score(self.x_test, self.y_test)
            if status_callback:
                status_callback(f"Training complete! Accuracy: {score*100:.2f}%")
            return self.model
        except Exception as e:
            if status_callback:
                status_callback(f"Training error: {e}")
            return None

    def evaluate_model(self):
        if not self.trained:
            return None
        score = self.model.score(self.x_test, self.y_test)
        return {"accuracy": score}

    def predict_digit(self, image_array):
        if not self.trained:
            return None
        flattened = image_array.reshape(1, -1)
        prediction = self.model.predict(flattened)
        return prediction[0]

    def save_model(self, filepath="digit_rf_model.pkl"):
        if not self.trained:
            return False
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_saved_model(self, filepath="digit_rf_model.pkl"):
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    self.model = pickle.load(f)
                self.trained = True
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            return False

class DrawingApp:
    def __init__(self, root, classifier):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("600x400")
        self.root.resizable(False, False)

        self.classifier = classifier
        self.setup_exception_handling()
        self.setup_ui()

    def setup_exception_handling(self):
        self.original_excepthook = sys.excepthook

        def custom_handler(exc_type, exc_value, exc_traceback):
            messagebox.showerror("Error", f"An error occurred: {exc_value}")
            self.original_excepthook(exc_type, exc_value, exc_traceback)

        sys.excepthook = custom_handler

    def setup_ui(self):
        drawing_frame = Frame(self.root, bd=2, relief="raised")
        drawing_frame.place(x=20, y=20, width=280, height=280)

        self.canvas = Canvas(drawing_frame, bg="black", width=280, height=280)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.setup_drawing()

        control_frame = Frame(self.root)
        control_frame.place(x=320, y=20, width=260, height=280)

        self.result_label = Label(control_frame, text="Draw a digit (0-9)", font=("Arial", 18))
        self.result_label.pack(pady=10)

        self.prediction_label = Label(control_frame, text="Prediction: None", font=("Arial", 24, "bold"))
        self.prediction_label.pack(pady=20)

        self.predict_btn = Button(control_frame, text="Predict", font=("Arial", 14), command=self.predict)
        self.predict_btn.pack(fill="x", pady=5)

        self.clear_btn = Button(control_frame, text="Clear Canvas", font=("Arial", 14), command=self.clear_canvas)
        self.clear_btn.pack(fill="x", pady=5)

        self.train_btn = Button(control_frame, text="Train Model", font=("Arial", 14), command=self.train_model_thread)
        self.train_btn.pack(fill="x", pady=5)

        self.load_btn = Button(control_frame, text="Load Model", font=("Arial", 14), command=self.load_model)
        self.load_btn.pack(fill="x", pady=5)

        self.status_var = tk.StringVar()
        self.status_var.set("Status: Ready")
        self.status_bar = Label(self.root, textvariable=self.status_var, bd=1, relief="sunken", anchor="w")
        self.status_bar.pack(side="bottom", fill="x")

    def setup_drawing(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 15
        self.color = "white"
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def paint(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                    width=self.line_width, fill=self.color, capstyle="round", smooth=True)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x = None
        self.old_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.prediction_label.config(text="Prediction: None")

    def preprocess_canvas(self):
        self.canvas.update()
        image = Image.new("L", (280, 280), "black")
        draw = ImageDraw.Draw(image)

        for item in self.canvas.find_all():
            if self.canvas.type(item) == "line":
                x1, y1, x2, y2 = self.canvas.coords(item)
                draw.line((x1, y1, x2, y2), fill="white", width=self.line_width)

        if hasattr(Image, 'Resampling'):
            resample_method = Image.Resampling.LANCZOS
        else:
            resample_method = Image.LANCZOS

        image = image.resize((28, 28), resample=resample_method)
        img_array = np.array(image).astype('float32') / 255.0
        return img_array

    def predict(self):
        if not self.classifier.trained:
            self.status_var.set("Status: Model not trained!")
            messagebox.showinfo("Model not ready", "Train or load the model first.")
            return

        img_array = self.preprocess_canvas()
        try:
            prediction = self.classifier.predict_digit(img_array)
            self.prediction_label.config(text=f"Prediction: {prediction}")
            self.status_var.set("Status: Prediction done.")
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Prediction Error", str(e))

    def update_status(self, message):
        self.status_var.set(f"Status: {message}")
        self.root.update_idletasks()

    def train_model_thread(self):
        self.update_status("Training model...")
        self.train_btn.config(state="disabled")

        def train_task():
            try:
                model = self.classifier.train_model(
                    status_callback=lambda msg: self.root.after(0, lambda: self.update_status(msg))
                )

                if model:
                    accuracy = self.classifier.evaluate_model()["accuracy"] * 100
                    if self.classifier.save_model():
                        self.root.after(0, lambda: self.update_status(f"Training complete! Accuracy: {accuracy:.2f}%. Model saved."))
                    else:
                        self.root.after(0, lambda: self.update_status(f"Training complete! Accuracy: {accuracy:.2f}%. Saving failed."))

            finally:
                self.root.after(0, lambda: self.train_btn.config(state="normal"))

        threading.Thread(target=train_task, daemon=True).start()

    def load_model(self):
        self.update_status("Loading model...")
        if self.classifier.load_saved_model():
            self.update_status("Model loaded successfully.")
            messagebox.showinfo("Success", "Model loaded successfully!")
        else:
            self.update_status("No saved model found.")
            if messagebox.askyesno("Model Not Found", "No saved model found. Would you like to train a new model now?"):
                self.train_model_thread()

def main():
    classifier = DigitClassifier()

    try:
        import numpy
        from PIL import Image
        import sklearn
    except ImportError as e:
        messagebox.showerror("Missing Dependency", f"{e}")
        return

    root = tk.Tk()
    app = DrawingApp(root, classifier)

    if not classifier.load_saved_model():
        app.status_var.set("Status: No model found. Please train one.")

    root.mainloop()

if __name__ == "__main__":
    main()
