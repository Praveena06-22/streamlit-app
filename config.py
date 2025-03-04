import os

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    
    # Upload configuration
    UPLOAD_FOLDER = os.path.join('static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Model paths
    MODELS_DIR = 'models'
    SOIL_CLASSIFIER_MODEL = os.path.join(MODELS_DIR, 'soil_classifier.joblib')
    RANDOM_FOREST_MODEL = os.path.join(MODELS_DIR, 'random_forest_model.joblib')
    DECISION_TREE_MODEL = os.path.join(MODELS_DIR, 'decision_tree_model.joblib')
    SCALER_MODEL = os.path.join(MODELS_DIR, 'scaler.joblib')
    CROP_ENCODER = os.path.join(MODELS_DIR, 'crop_encoder.joblib')
    SOIL_TYPE_ENCODER = os.path.join(MODELS_DIR, 'soil_type_encoder.joblib')
    
    # Dataset paths
    DATA_DIR = 'data'
    CROP_DATASET = os.path.join(DATA_DIR, 'crop_dataset.csv')
    SOIL_TYPE_DATASET = os.path.join(os.getcwd(), DATA_DIR, 'soil_type_dataset')
    
    # Model parameters
    SOIL_IMAGE_SIZE = (200, 200)  # Size to resize soil images for analysis
    N_RECOMMENDATIONS = 1  # Number of crop recommendations to provide
    
    # Soil types and their characteristics
    SOIL_TYPES = {
        'alluvial': {
            'properties': {
                'texture': 'Fine, fertile particles',
                'water_retention': 'Good',
                'drainage': 'Good',
                'fertility': 'High'
            },
            'suitable_crops': ['Rice', 'Wheat', 'Sugarcane', 'Maize']
        },
        'clayey': {
            'properties': {
                'texture': 'Fine, dense particles',
                'water_retention': 'High',
                'drainage': 'Poor',
                'fertility': 'Medium to High'
            },
            'suitable_crops': ['Rice', 'Cotton', 'Wheat', 'Pulses']
        },
        'sandy': {
            'properties': {
                'texture': 'Coarse particles',
                'water_retention': 'Low',
                'drainage': 'Excellent',
                'fertility': 'Low'
            },
            'suitable_crops': ['Groundnut', 'Potato', 'Carrot', 'Watermelon']
        },
        'sandy_loam': {
            'properties': {
                'texture': 'Mixed particle sizes',
                'water_retention': 'Medium',
                'drainage': 'Good',
                'fertility': 'Medium'
            },
            'suitable_crops': ['Vegetables', 'Fruits', 'Tobacco', 'Cotton']
        }
    }
    
    # Parameter ranges for validation
    PARAMETER_RANGES = {
        'N': (0, 140),    # Nitrogen (mg/kg)
        'P': (0, 145),    # Phosphorus (mg/kg)
        'K': (0, 205),    # Potassium (mg/kg)
        'pH': (0, 14),    # pH scale
        'temperature': (-20, 50),   # Temperature (°C)
        'rainfall': (0, 5000)       # Annual rainfall (mm)
    }
    
    @staticmethod
    def init_app(app):
        """Initialize application configuration"""
        # Create required directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        
        # Set Flask configuration
        app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
        
    @staticmethod
    def allowed_file(filename):
        """Check if a filename has an allowed extension"""
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
            
    @staticmethod
    def validate_parameters(parameters):
        """Validate input parameters against defined ranges"""
        for param, value in parameters.items():
            if param in Config.PARAMETER_RANGES:
                min_val, max_val = Config.PARAMETER_RANGES[param]
                if not min_val <= float(value) <= max_val:
                    raise ValueError(
                        f"{param} value {value} is outside valid range ({min_val}, {max_val})"
                    )
