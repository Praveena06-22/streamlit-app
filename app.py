import os
import streamlit as st
from werkzeug.utils import secure_filename
from soil_classifier import SoilClassifier
from config import Config
from train_models import predict_crop

# Initialize configuration
Config.init_app(None)

# Initialize soil classifier
soil_classifier = SoilClassifier()
soil_classifier.train(Config.SOIL_TYPE_DATASET)

# Test credentials (in a real app, use a database)
USERS = {
    'test_user': 'test_password'
}

# Session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Streamlit login function
def login():
    if st.session_state['logged_in']:
        st.success('You are already logged in!')
        dashboard()
        return
    
    st.title('Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    
    if st.button('Submit'):
        if username in USERS and USERS[username] == password:
            st.session_state['logged_in'] = True
            st.success('Login successful!')
            dashboard()
        else:
            st.error('Invalid credentials!')

# Dashboard
def dashboard():
    if not st.session_state['logged_in']:
        st.error('Please log in first.')
        login()
        return

    st.title('Dashboard')
    st.markdown('Welcome to the Soil and Crop Recommendation App.')
    
    menu = ['Soil Analysis', 'Crop Recommendation', 'Logout']
    choice = st.sidebar.selectbox('Select an option:', menu)

    if choice == 'Soil Analysis':
        soil_analysis()
    elif choice == 'Crop Recommendation':
        crop_recommendation()
    elif choice == 'Logout':
        st.session_state['logged_in'] = False
        st.success('You have logged out successfully!')
        login()

# Soil Analysis Function
def soil_analysis():
    if not st.session_state['logged_in']:
        st.error('Please log in first.')
        login()
        return

    st.title('Soil Analysis')
    
    uploaded_file = st.file_uploader("Choose a soil image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        filename = secure_filename(uploaded_file.name)
        filepath = os.path.join("/tmp", filename)
        with open(filepath, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Analyze soil
        try:
            result = soil_classifier.predict(filepath)
            soil_type_key = result['soil_type'].lower().replace(' ', '_')
            soil_info = Config.SOIL_TYPES.get(soil_type_key, {'properties': {}, 'suitable_crops': []})
            
            st.image(filepath, caption="Uploaded Image", use_column_width=True)
            st.write(f"Soil Type: {result['soil_type']}")
            st.write(f"Confidence: {result['confidence']}")
            st.write(f"Properties: {soil_info['properties']}")
            st.write(f"Recommended Crops: {soil_info['suitable_crops']}")
        except Exception as e:
            st.error(f'Error analyzing soil: {str(e)}')

# Crop Recommendation Function
def crop_recommendation():
    if not st.session_state['logged_in']:
        st.error('Please log in first.')
        login()
        return

    st.title('Crop Recommendation')

    nitrogen = st.number_input('Nitrogen (N)', min_value=0.0)
    phosphorus = st.number_input('Phosphorus (P)', min_value=0.0)
    potassium = st.number_input('Potassium (K)', min_value=0.0)
    ph = st.number_input('pH Level', min_value=0.0)
    temperature = st.number_input('Temperature (°C)', min_value=0.0)
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0)

    if st.button('Recommend Crops'):
        try:
            features = {
                'N': nitrogen,
                'P': phosphorus,
                'K': potassium,
                'pH': ph,
                'temperature': temperature,
                'rainfall': rainfall
            }

            # Validate parameters
            Config.validate_parameters(features)

            # Get crop recommendations
            recommended_crops = predict_crop(features, Config.N_RECOMMENDATIONS)

            st.write(f"Recommended Crops: {recommended_crops}")
        except ValueError as e:
            st.error(f'Invalid input: {str(e)}')
        except Exception as e:
            st.error(f'Error processing request: {str(e)}')

# Main Function to Run the App
if not st.session_state['logged_in']:
    login()
else:
    dashboard()
