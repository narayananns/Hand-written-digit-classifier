import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import pickle
import os
import cv2
from streamlit_drawable_canvas import st_canvas
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Advanced Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvancedDigitClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.trained = False
        self.data_loaded = False
        self.model_type = "RandomForest"
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def extract_features(self, X):
        """Extract advanced features from digit images"""
        features = []
        
        for img in X:
            # Reshape to 28x28 if it's flattened
            if img.shape == (784,):
                img_2d = img.reshape(28, 28)
            else:
                img_2d = img
            
            # Original pixel values
            pixel_features = img_2d.flatten()
            
            # Histogram features
            hist, _ = np.histogram(img_2d, bins=10, range=(0, 1))
            hist_features = hist / np.sum(hist) if np.sum(hist) > 0 else hist
            
            # Moments (center of mass, etc.)
            y_coords, x_coords = np.mgrid[0:28, 0:28]
            total_intensity = np.sum(img_2d)
            
            if total_intensity > 0:
                center_x = np.sum(x_coords * img_2d) / total_intensity
                center_y = np.sum(y_coords * img_2d) / total_intensity
                
                # Variance around center
                var_x = np.sum((x_coords - center_x)**2 * img_2d) / total_intensity
                var_y = np.sum((y_coords - center_y)**2 * img_2d) / total_intensity
                
                moment_features = [center_x/28, center_y/28, var_x/784, var_y/784]
            else:
                moment_features = [0, 0, 0, 0]
            
            # Projection features
            h_projection = np.sum(img_2d, axis=1)  # Horizontal projection
            v_projection = np.sum(img_2d, axis=0)  # Vertical projection
            
            # Normalize projections
            h_projection = h_projection / np.max(h_projection) if np.max(h_projection) > 0 else h_projection
            v_projection = v_projection / np.max(v_projection) if np.max(v_projection) > 0 else v_projection
            
            # Zoning features (divide image into 4x4 zones)
            zone_features = []
            for i in range(0, 28, 7):
                for j in range(0, 28, 7):
                    zone = img_2d[i:i+7, j:j+7]
                    zone_features.append(np.sum(zone))
            
            # Combine all features
            combined_features = np.concatenate([
                pixel_features,
                hist_features,
                moment_features,
                h_projection,
                v_projection,
                zone_features
            ])
            
            features.append(combined_features)
        
        return np.array(features)

    def augment_data(self, X, y, augmentation_factor=2):
        """Augment training data with rotations and translations"""
        augmented_X = []
        augmented_y = []
        
        for i, (img, label) in enumerate(zip(X, y)):
            if i % 1000 == 0:
                st.info(f"Augmenting data: {i}/{len(X)}")
            
            # Original image
            augmented_X.append(img)
            augmented_y.append(label)
            
            # Reshape to 28x28 for augmentation
            if img.shape == (784,):
                img_2d = img.reshape(28, 28)
            else:
                img_2d = img
            
            # Convert to PIL Image for easier manipulation
            pil_img = Image.fromarray((img_2d * 255).astype(np.uint8))
            
            for _ in range(augmentation_factor):
                # Random rotation (-15 to 15 degrees)
                angle = np.random.uniform(-15, 15)
                rotated = pil_img.rotate(angle, fillcolor=0)
                
                # Random translation
                dx = np.random.randint(-2, 3)
                dy = np.random.randint(-2, 3)
                translated = rotated.transform((28, 28), Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=0)
                
                # Convert back to array
                aug_img = np.array(translated) / 255.0
                augmented_X.append(aug_img.flatten())
                augmented_y.append(label)
        
        return np.array(augmented_X), np.array(augmented_y)

    def build_model(self, model_type="RandomForest"):
        """Build different types of models"""
        self.model_type = model_type
        
        if model_type == "RandomForest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            )
        elif model_type == "SVM":
            self.model = SVC(
                C=10,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            )
        elif model_type == "MLP":
            self.model = MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                max_iter=500,
                alpha=0.001,
                learning_rate_init=0.001,
                random_state=42
            )
        
        return self.model

    def load_data(self, use_augmentation=True):
        """Load and preprocess data"""
        try:
            st.info("Loading MNIST dataset...")
            
            try:
                from sklearn.datasets import fetch_openml
                mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
                X = mnist.data.astype('float32') / 255.0
                y = mnist.target.astype('int')
                
                # Use subset for faster training in demo
                if len(X) > 20000:
                    indices = np.random.choice(len(X), 20000, replace=False)
                    X = X[indices]
                    y = y[indices]
                    
            except Exception:
                st.warning("Failed to load MNIST. Loading smaller digits dataset...")
                from sklearn.datasets import load_digits
                digits = load_digits()
                X = digits.data / 16.0
                y = digits.target

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Data augmentation for training set
            if use_augmentation and len(X_train) < 50000:
                st.info("Applying data augmentation...")
                X_train, y_train = self.augment_data(X_train, y_train, augmentation_factor=1)
            
            # Feature extraction
            st.info("Extracting advanced features...")
            X_train_features = self.extract_features(X_train)
            X_test_features = self.extract_features(X_test)
            
            # Normalize features
            X_train_scaled = self.scaler.fit_transform(X_train_features)
            X_test_scaled = self.scaler.transform(X_test_features)
            
            self.x_train = X_train_scaled
            self.x_test = X_test_scaled
            self.y_train = y_train
            self.y_test = y_test
            self.data_loaded = True
            
            st.success(f"Dataset loaded successfully! Training samples: {len(X_train_scaled)}")
            return (self.x_train, self.y_train), (self.x_test, self.y_test)

        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    def train_model(self, model_type="RandomForest", use_grid_search=False):
        """Train the model with optional hyperparameter tuning"""
        if not self.data_loaded:
            if self.load_data() is None:
                return None

        try:
            st.info(f"Training {model_type} model...")
            
            if use_grid_search and model_type == "RandomForest":
                st.info("Performing hyperparameter tuning...")
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [15, 20, 25],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                
                rf = RandomForestClassifier(random_state=42, n_jobs=-1)
                grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                grid_search.fit(self.x_train, self.y_train)
                self.model = grid_search.best_estimator_
                
                st.success(f"Best parameters: {grid_search.best_params_}")
            else:
                self.model = self.build_model(model_type)
                self.model.fit(self.x_train, self.y_train)
            
            self.trained = True
            self.model_type = model_type
            
            # Evaluate model
            train_score = self.model.score(self.x_train, self.y_train)
            test_score = self.model.score(self.x_test, self.y_test)
            
            st.success(f"Training complete!")
            st.info(f"Training Accuracy: {train_score*100:.2f}%")
            st.info(f"Test Accuracy: {test_score*100:.2f}%")
            
            return self.model
            
        except Exception as e:
            st.error(f"Training error: {e}")
            return None

    def evaluate_model(self):
        """Comprehensive model evaluation"""
        if not self.trained:
            return None
        
        y_pred = self.model.predict(self.x_test)
        train_score = self.model.score(self.x_train, self.y_train)
        test_score = self.model.score(self.x_test, self.y_test)
        
        # Classification report
        report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        return {
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "classification_report": report,
            "confusion_matrix": cm
        }

    def preprocess_drawn_image(self, image_array):
        """Advanced preprocessing for drawn images"""
        # Convert to PIL Image
        if image_array.max() <= 1.0:
            pil_img = Image.fromarray((image_array * 255).astype(np.uint8))
        else:
            pil_img = Image.fromarray(image_array.astype(np.uint8))
        
        # Apply Gaussian blur to smooth the image
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Center the digit
        bbox = pil_img.getbbox()
        if bbox:
            # Crop to bounding box
            cropped = pil_img.crop(bbox)
            
            # Calculate size to maintain aspect ratio
            width, height = cropped.size
            max_size = max(width, height)
            
            # Create a square image with padding
            new_img = Image.new('L', (max_size, max_size), 0)
            paste_x = (max_size - width) // 2
            paste_y = (max_size - height) // 2
            new_img.paste(cropped, (paste_x, paste_y))
            
            # Resize to 28x28
            final_img = new_img.resize((28, 28), Image.Resampling.LANCZOS)
        else:
            final_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(final_img).astype('float32') / 255.0
        
        return img_array

    def predict_digit(self, image_array):
        """Predict digit with confidence scores"""
        if not self.trained:
            return None, None
        
        # Preprocess the image
        processed_img = self.preprocess_drawn_image(image_array)
        
        # Extract features
        features = self.extract_features(processed_img.reshape(1, -1))
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(scaled_features)[0]
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(scaled_features)[0]
        else:
            # For SVM without probability=True
            probabilities = np.zeros(10)
            probabilities[prediction] = 1.0
        
        return prediction, probabilities

    def save_model(self, filepath="advanced_digit_model.pkl"):
        """Save model with scaler"""
        if not self.trained:
            return False
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            return True
        except Exception as e:
            st.error(f"Error saving model: {e}")
            return False

    def load_saved_model(self, filepath="advanced_digit_model.pkl"):
        """Load model with scaler"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.model_type = model_data.get('model_type', 'Unknown')
                self.trained = True
                return True
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return False
        return False

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = AdvancedDigitClassifier()

def main():
    st.title("🔢 Advanced Handwritten Digit Recognition")
    st.markdown("Enhanced AI model with feature engineering and data augmentation!")
    
    # Sidebar for model controls
    with st.sidebar:
        st.header("🎛️ Model Controls")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model Type:",
            ["RandomForest", "SVM", "MLP"],
            index=0
        )
        
        # Model status
        if st.session_state.classifier.trained:
            st.success(f"✅ {st.session_state.classifier.model_type} model ready!")
        else:
            st.warning("⚠️ Model not trained")
        
        # Training options
        st.subheader("Training Options")
        use_grid_search = st.checkbox("Use Grid Search (slower but better)", value=False)
        
        # Load model button
        if st.button("Load Saved Model"):
            if st.session_state.classifier.load_saved_model():
                st.success("Model loaded successfully!")
                st.rerun()
            else:
                st.error("No saved model found.")
        
        # Train model button
        if st.button("Train New Model"):
            if st.session_state.classifier.train_model(model_type, use_grid_search):
                if st.session_state.classifier.save_model():
                    st.success("Model trained and saved!")
                else:
                    st.warning("Model trained but saving failed.")
                st.rerun()
        
        # Model evaluation
        if st.session_state.classifier.trained:
            st.subheader("📊 Model Performance")
            eval_results = st.session_state.classifier.evaluate_model()
            if eval_results:
                st.metric("Training Accuracy", f"{eval_results['train_accuracy']*100:.2f}%")
                st.metric("Test Accuracy", f"{eval_results['test_accuracy']*100:.2f}%")
                
                # Show confusion matrix
                if st.checkbox("Show Confusion Matrix"):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(eval_results['confusion_matrix'], 
                              annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title('Confusion Matrix')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✏️ Draw a Digit")
        st.markdown("Draw clearly and center the digit for best results!")
        
        # Canvas for drawing
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            background_image=None,
            update_streamlit=True,
            height=300,
            width=300,
            drawing_mode="freedraw",
            point_display_radius=0,
            key="canvas",
        )
        
        # Control buttons
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)
        with col1_2:
            if st.button("🗑️ Clear Canvas", use_container_width=True):
                st.rerun()
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        # Prediction logic
        if predict_btn and canvas_result.image_data is not None:
            if not st.session_state.classifier.trained:
                st.error("Please train or load a model first!")
            else:
                # Convert canvas to grayscale
                image_data = canvas_result.image_data[:, :, 0]  # Take only one channel
                
                # Check if image has content
                if np.sum(image_data) > 0:
                    try:
                        # Make prediction
                        prediction, probabilities = st.session_state.classifier.predict_digit(image_data)
                        
                        if prediction is not None:
                            # Display prediction
                            st.markdown(f"### Predicted Digit: **{prediction}**")
                            
                            # Display confidence
                            confidence = probabilities[prediction] * 100
                            st.markdown(f"**Confidence:** {confidence:.1f}%")
                            
                            # Progress bar for confidence
                            st.progress(probabilities[prediction])
                            
                            # Show probability distribution
                            st.subheader("📊 Probability Distribution")
                            prob_df = {
                                'Digit': list(range(10)),
                                'Probability': probabilities * 100
                            }
                            st.bar_chart(prob_df, x='Digit', y='Probability')
                            
                            # Show top 3 predictions
                            st.subheader("🏆 Top 3 Predictions")
                            top_3_indices = np.argsort(probabilities)[-3:][::-1]
                            for i, idx in enumerate(top_3_indices):
                                st.write(f"{i+1}. **Digit {idx}**: {probabilities[idx]*100:.1f}%")
                        
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                else:
                    st.warning("Please draw something on the canvas!")
        
        # Show processed image preview
        if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
            st.subheader("🔍 Processed Image")
            image_data = canvas_result.image_data[:, :, 0]
            processed_img = st.session_state.classifier.preprocess_drawn_image(image_data)
            st.image(processed_img, width=100, caption="28x28 processed")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Advanced ML techniques")
    
    # Instructions
    with st.expander("📖 How to use this Advanced System"):
        st.markdown("""
        ## 🚀 Features:
        - **Advanced Feature Engineering**: Extracts histograms, moments, projections, and zoning features
        - **Data Augmentation**: Increases training data with rotations and translations
        - **Multiple Model Types**: Choose between RandomForest, SVM, and MLP
        - **Hyperparameter Tuning**: Optional grid search for optimal parameters
        - **Smart Preprocessing**: Centers and normalizes drawn digits
        
        ## 📝 Instructions:
        1. **Select Model**: Choose your preferred model type from the sidebar
        2. **Train/Load**: Train a new model or load a saved one
        3. **Draw**: Draw a digit clearly and centered on the canvas
        4. **Predict**: Click predict to get AI analysis with confidence scores
        
        ## 💡 Tips for Best Results:
        - Draw digits **clearly** and **centered**
        - Use **consistent stroke width**
        - Make digits **large enough** to fill most of the canvas
        - Try different models to see which works best for your drawing style
        """)

if __name__ == "__main__":
    main()