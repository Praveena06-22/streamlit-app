import requests

def test_crop_recommendation():
    # Server URL
    base_url = 'http://localhost:3001'
    
    # First, login to get a session
    login_data = {
        'username': 'test_user',
        'password': 'test_password'
    }
    
    session = requests.Session()
    
    # Login
    login_response = session.post(f'{base_url}/login', data=login_data)
    if login_response.status_code != 200:
        print("Login failed!")
        return
        
    # Test crop recommendation with sample data within valid ranges
    # Note: Field names must match the form field names in app.py and feature names in train_models.py
    crop_data = {
        'N': '90',      # Good middle value for most crops
        'P': '45',    # Middle range value
        'K': '60',     # Middle range value
        'pH': '6.5',          # Neutral pH, good for most crops
        'temperature': '25',   # Moderate temperature
        'rainfall': '150'      # Moderate rainfall
    }
    
    response = session.post(f'{base_url}/crop-recommendation', data=crop_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    test_crop_recommendation()
