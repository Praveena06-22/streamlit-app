# Smart Crop Recommendation System

A machine learning-based system that recommends suitable crops based on soil analysis and weather conditions. The system uses computer vision to analyze soil images and combines that with soil content data and weather parameters to provide accurate crop recommendations.

## Features

- User authentication system
- Soil type classification through image analysis
- Crop recommendation based on:
  - Soil NPK values
  - pH levels
  - Temperature
  - Rainfall
- Advanced machine learning models:
  - Random Forest Classifier
  - Decision Tree Classifier
  - Computer Vision for soil analysis
- Detailed recommendations with:
  - Confidence scores
  - Expected yield estimates
  - Growing season information

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crop-recommendation-system.git
cd crop-recommendation-system
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Generate the dataset and train models:
```bash
python generate_dataset.py
python train_models.py
```

5. Run the application:
```bash
python app.py
```

6. Access the application:
Open your web browser and navigate to `http://localhost:3001`

## Usage Guide

1. **Login**
   - Use the provided test credentials:
     - Username: test_user
     - Password: test_password

2. **Soil Analysis**
   - Navigate to the Soil Analysis page
   - Upload a clear image of your soil sample
   - The system will analyze and classify the soil type
   - View detailed soil properties and suitable crops

3. **Crop Recommendation**
   - Enter soil content values (N, P, K, pH)
   - Input weather conditions (temperature, rainfall)
   - Submit to get personalized crop recommendations
   - View detailed recommendations including:
     - Crop suitability scores
     - Expected yields
     - Growing seasons

## Model Information

### Soil Classification
- Uses computer vision techniques
- Trained on a dataset of soil images
- Classifies into main soil types:
  - Alluvial
  - Clayey
  - Sandy
  - Sandy Loam

### Crop Recommendation
- Uses ensemble of machine learning models
- Features considered:
  - Nitrogen content (N)
  - Phosphorus content (P)
  - Potassium content (K)
  - pH value
  - Temperature
  - Rainfall
- Provides recommendations with confidence scores

## Directory Structure

```
crop_recommendation_system/
├── app.py                 # Main Flask application
├── generate_dataset.py    # Dataset generation script
├── train_models.py        # Model training script
├── soil_classifier.py     # Soil classification module
├── requirements.txt       # Project dependencies
├── static/               # Static files (CSS, uploads)
├── templates/            # HTML templates
├── models/              # Trained model files
└── data/                # Dataset files
    └── soil_type_dataset/  # Soil images for training
```

## Technical Details

- **Framework**: Flask
- **ML Libraries**: scikit-learn, OpenCV
- **Frontend**: HTML, CSS
- **Database**: SQLite (for user authentication)
- **Image Processing**: OpenCV, PIL

## Model Performance

- Soil Classification Accuracy: ~85%
- Crop Recommendation Accuracy: ~90%
- Regular model updates based on new data

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
