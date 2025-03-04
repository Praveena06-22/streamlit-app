import pandas as pd
import numpy as np

# Define crop characteristics
CROPS = {
    'rice': {
        'n': (60, 120),  # Nitrogen range (min, max)
        'p': (20, 60),   # Phosphorus range
        'k': (30, 70),   # Potassium range
        'ph': (5.5, 7.5),
        'temperature': (20, 35),
        'rainfall': (150, 300),
        'season': 'Kharif',
        'yield_range': (3.0, 6.0)  # Tons per hectare
    },
    'wheat': {
        'n': (100, 140),
        'p': (40, 80),
        'k': (40, 90),
        'ph': (6.0, 7.5),
        'temperature': (15, 25),
        'rainfall': (75, 150),
        'season': 'Rabi',
        'yield_range': (2.5, 5.0)
    },
    'cotton': {
        'n': (80, 120),
        'p': (30, 60),
        'k': (50, 80),
        'ph': (5.5, 8.0),
        'temperature': (20, 30),
        'rainfall': (100, 200),
        'season': 'Kharif',
        'yield_range': (1.5, 3.0)
    },
    'sugarcane': {
        'n': (100, 140),
        'p': (50, 90),
        'k': (80, 120),
        'ph': (6.0, 7.5),
        'temperature': (20, 35),
        'rainfall': (150, 300),
        'season': 'Year-round',
        'yield_range': (60, 100)
    },
    'groundnut': {
        'n': (40, 80),
        'p': (30, 60),
        'k': (30, 60),
        'ph': (6.0, 7.0),
        'temperature': (25, 35),
        'rainfall': (100, 200),
        'season': 'Kharif',
        'yield_range': (1.5, 3.0)
    },
    'maize': {
        'n': (80, 120),
        'p': (40, 80),
        'k': (40, 80),
        'ph': (5.5, 7.5),
        'temperature': (20, 30),
        'rainfall': (100, 200),
        'season': 'Kharif',
        'yield_range': (4.0, 8.0)
    },
    'pulses': {
        'n': (20, 60),
        'p': (40, 80),
        'k': (30, 60),
        'ph': (6.0, 7.5),
        'temperature': (20, 30),
        'rainfall': (60, 150),
        'season': 'Rabi',
        'yield_range': (1.0, 2.5)
    },
    'tomato': {
        'n': (60, 120),
        'p': (40, 80),
        'k': (40, 80),
        'ph': (6.0, 7.0),
        'temperature': (20, 30),
        'rainfall': (60, 150),
        'season': 'Year-round',
        'yield_range': (20, 40)
    },
    'cabbage': {
        'n': (80, 120),
        'p': (40, 80),
        'k': (40, 80),
        'ph': (6.0, 7.0),
        'temperature': (15, 25),
        'rainfall': (75, 150),
        'season': 'Winter',
        'yield_range': (25, 45)
    },
    'cauliflower': {
        'n': (80, 120),
        'p': (60, 90),
        'k': (40, 80),
        'ph': (6.0, 7.0),
        'temperature': (15, 25),
        'rainfall': (75, 150),
        'season': 'Winter',
        'yield_range': (20, 35)
    },
    'watermelon': {
        'n': (60, 100),
        'p': (40, 70),
        'k': (50, 90),
        'ph': (6.0, 7.0),
        'temperature': (25, 35),
        'rainfall': (100, 150),
        'season': 'Summer',
        'yield_range': (25, 40)
    },
    'potato': {
        'n': (100, 140),
        'p': (50, 80),
        'k': (60, 100),
        'ph': (5.0, 6.5),
        'temperature': (15, 25),
        'rainfall': (50, 150),
        'season': 'Rabi',
        'yield_range': (20, 35)
    },
    'tobacco': {
        'n': (60, 100),
        'p': (20, 40),
        'k': (40, 60),
        'ph': (5.5, 6.5),
        'temperature': (20, 30),
        'rainfall': (50, 125),
        'season': 'Kharif',
        'yield_range': (1.5, 2.5)
    }
}

def generate_dataset(num_samples=1000):
    np.random.seed(42)  # For reproducibility
    
    # Soil types from the image dataset
    soil_types = ['alluvial', 'clayey', 'sandy', 'sandy loam']
    
    # Generate random data
    data = {
        'Nitrogen': np.random.uniform(0, 140, num_samples),  # N (kg/ha)
        'Phosphorus': np.random.uniform(5, 145, num_samples),  # P (kg/ha)
        'Potassium': np.random.uniform(5, 205, num_samples),  # K (kg/ha)
        'pH': np.random.uniform(3.5, 10.0, num_samples),
        'Temperature': np.random.uniform(8.8, 43.7, num_samples),  # °C
        'Rainfall': np.random.uniform(20.2, 298.6, num_samples),  # mm
        'Soil_Type': np.random.choice(soil_types, num_samples)
    }
    
    # Function to determine suitable crop based on conditions
    def get_suitable_crop(row):
        if row['Soil_Type'] == 'alluvial':
            if row['Rainfall'] > 200 and row['Temperature'] > 20:
                return np.random.choice(['rice', 'sugarcane', 'wheat', 'tomato'])
            return np.random.choice(['wheat', 'cabbage', 'cauliflower'])
            
        elif row['Soil_Type'] == 'clayey':
            if row['Rainfall'] > 150:
                return np.random.choice(['rice', 'cotton', 'sugarcane'])
            return np.random.choice(['wheat', 'cotton'])
            
        elif row['Soil_Type'] == 'sandy':
            if row['Temperature'] > 25:
                return np.random.choice(['groundnut', 'potato', 'watermelon'])
            return np.random.choice(['carrot', 'potato'])
            
        else:  # sandy loam
            if row['Rainfall'] > 100 and row['Temperature'] > 20:
                return np.random.choice(['tomato', 'maize', 'cabbage'])
            return np.random.choice(['cauliflower', 'tobacco'])
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add crop recommendations based on conditions
    df['Crop'] = df.apply(get_suitable_crop, axis=1)
    
    # Save dataset
    df.to_csv('data/crop_dataset.csv', index=False)
    print(f"Dataset generated with {num_samples} samples and saved to crop_dataset.csv")
    
    # Display sample statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    print(df.describe())
    print("\nSoil Type Distribution:")
    print(df['Soil_Type'].value_counts())
    print("\nCrop Distribution:")
    print(df['Crop'].value_counts())

if __name__ == "__main__":
    generate_dataset(2000)  # Generate 2000 samples
