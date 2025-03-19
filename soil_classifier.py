import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from skimage.feature import greycomatrix, greycoprops

class SoilClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.image_size = (200, 200)  # Standard size for all images
        
    def _load_image(self, image_path):
        """Load and preprocess image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        img = cv2.resize(img, self.image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _load_images(self, dataset_path):
        """Load images and labels from the dataset directory"""
        images = []
        labels = []
        
        for soil_type in os.listdir(dataset_path):
            soil_path = os.path.join(dataset_path, soil_type)
            if os.path.isdir(soil_path):
                for image_file in os.listdir(soil_path):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(soil_path, image_file)
                        try:
                            img = self._load_image(image_path)
                            features = self._extract_features(img)
                            images.append(features)
                            labels.append(soil_type)
                        except Exception as e:
                            print(f"Error loading image {image_path}: {str(e)}")
        
        return np.array(images), np.array(labels)
    
    def _extract_features(self, image):
        """Extract features from the image"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Calculate color histograms
        hist_rgb = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist_lab = cv2.calcHist([lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # Normalize histograms
        hist_rgb = cv2.normalize(hist_rgb, hist_rgb).flatten()
        hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
        hist_lab = cv2.normalize(hist_lab, hist_lab).flatten()
        
        # Calculate texture features using GLCM
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        texture_features = self._calculate_texture_features(gray)
        
        # Combine all features
        features = np.concatenate([hist_rgb, hist_hsv, hist_lab, texture_features])
        return features
    
    def _calculate_texture_features(self, gray_image):
        """Calculate texture features using GLCM"""
        # Calculate GLCM
        glcm = greycomatrix(gray_image, [1], [0], symmetric=True, normed=True)
        
        # Calculate texture properties
        contrast = greycoprops(glcm, 'contrast')[0, 0]
        correlation = greycoprops(glcm, 'correlation')[0, 0]
        energy = greycoprops(glcm, 'energy')[0, 0]
        homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
        
        return np.array([contrast, correlation, energy, homogeneity])
    
    def train(self, dataset_path):
        """Train the soil classifier"""
        print(f"Loading images from: {dataset_path}")  # Debugging line
        X, y = self._load_images(dataset_path)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        accuracy = self.model.score(X_test, y_test)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        
        # Save model and encoder
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/soil_classifier.joblib')
        joblib.dump(self.label_encoder, 'models/soil_type_encoder.joblib')
    
    def predict(self, image_path):
        """Predict soil type from an image"""
        try:
            # Load model and encoder if not already loaded
            if self.model is None:
                self.model = joblib.load('models/soil_classifier.joblib')
                self.label_encoder = joblib.load('models/soil_type_encoder.joblib')
            
            # Load and preprocess image
            img = self._load_image(image_path)
            
            # Extract features
            features = self._extract_features(img)
            
            # Get prediction and probability
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            confidence = probabilities[prediction]
            
            # Get soil type name
            soil_type = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get soil properties from config
            from config import Config
            soil_info = Config.SOIL_TYPES.get(soil_type.lower().replace(' ', '_'), {})
            
            return {
                'soil_type': soil_type,
                'confidence': confidence,
                'properties': soil_info.get('properties', {}),
                'suitable_crops': soil_info.get('suitable_crops', [])
            }
            
        except Exception as e:
            raise Exception(f"Error predicting soil type: {str(e)}")
