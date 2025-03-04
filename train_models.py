import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_models():
    """Train and save the crop recommendation models"""
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv('data/crop_dataset.csv')
    
    # Prepare features and target
    X = df[['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Temperature', 'Rainfall']]
    y = df['Crop']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode crop labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train_encoded)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test_encoded, rf_pred)
    
    # Train Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_scaled, y_train_encoded)
    dt_pred = dt_model.predict(X_test_scaled)
    dt_accuracy = accuracy_score(y_test_encoded, dt_pred)
    
    # Print model performance
    print("\nRandom Forest Model Performance:")
    print(f"Accuracy: {rf_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, rf_pred, 
                              target_names=label_encoder.classes_))
    
    print("\nDecision Tree Model Performance:")
    print(f"Accuracy: {dt_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, dt_pred, 
                              target_names=label_encoder.classes_))
    
    # Save models and preprocessing objects
    joblib.dump(rf_model, 'models/random_forest_model.joblib')
    joblib.dump(dt_model, 'models/decision_tree_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(label_encoder, 'models/crop_encoder.joblib')
    
    return {
        'random_forest_accuracy': rf_accuracy,
        'decision_tree_accuracy': dt_accuracy,
        'label_encoder': label_encoder,
        'feature_names': X.columns.tolist()
    }

def predict_crop(features, n_recommendations=3):
    """
    Predict the best crops based on input features
    
    Args:
        features (dict): Dictionary containing N, P, K, pH, temperature, and rainfall values
        n_recommendations (int): Number of crop recommendations to return
    
    Returns:
        list: List of dictionaries containing crop recommendations with confidence scores
    """
    # Load models and preprocessing objects
    rf_model = joblib.load('models/random_forest_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    label_encoder = joblib.load('models/crop_encoder.joblib')
    
    # Prepare input features
    feature_names = ['N', 'P', 'K', 'pH', 'temperature', 'rainfall']
    X = np.array([[features[name] for name in feature_names]])
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get probability predictions
    probabilities = rf_model.predict_proba(X_scaled)[0]
    
    # Get top N recommendations
    top_indices = np.argsort(probabilities)[-n_recommendations:][::-1]
    
    recommendations = []
    for idx in top_indices:
        crop_name = label_encoder.inverse_transform([idx])[0]
        confidence = probabilities[idx]
        
        # Add recommendation details
        recommendations.append({
            'name': crop_name,
            'confidence': confidence,
            'season': get_crop_season(crop_name),
            'yield_estimate': estimate_yield(crop_name, features)
        })
    
    return recommendations

def get_crop_season(crop_name):
    """Get the growing season for a crop"""
    from generate_dataset import CROPS
    return CROPS[crop_name.lower()]['season']

def estimate_yield(crop_name, features):
    """Estimate crop yield based on conditions"""
    from generate_dataset import CROPS
    
    crop_info = CROPS[crop_name.lower()]
    
    # Calculate how optimal the conditions are
    conditions = [
        (features['N'] - np.mean(crop_info['n'])) / (crop_info['n'][1] - crop_info['n'][0]),
        (features['P'] - np.mean(crop_info['p'])) / (crop_info['p'][1] - crop_info['p'][0]),
        (features['K'] - np.mean(crop_info['k'])) / (crop_info['k'][1] - crop_info['k'][0]),
        (features['pH'] - np.mean(crop_info['ph'])) / (crop_info['ph'][1] - crop_info['ph'][0]),
        (features['temperature'] - np.mean(crop_info['temperature'])) / (crop_info['temperature'][1] - crop_info['temperature'][0]),
        (features['rainfall'] - np.mean(crop_info['rainfall'])) / (crop_info['rainfall'][1] - crop_info['rainfall'][0])
    ]
    
    optimality = 1 - np.mean(np.abs(conditions))
    yield_range = crop_info['yield_range']
    expected_yield = yield_range[0] + optimality * (yield_range[1] - yield_range[0])
    
    return round(expected_yield, 2)

if __name__ == '__main__':
    # Generate dataset if it doesn't exist
    if not os.path.exists('data/crop_dataset.csv'):
        print("Generating dataset...")
        from generate_dataset import generate_dataset
        df = generate_dataset()
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/crop_dataset.csv', index=False)
    
    # Train models
    print("Training models...")
    results = train_models()
    
    print("\nModel training completed!")
    print(f"Random Forest Accuracy: {results['random_forest_accuracy']:.4f}")
    print(f"Decision Tree Accuracy: {results['decision_tree_accuracy']:.4f}")
    print("\nAvailable crops:", results['label_encoder'].classes_.tolist())
