import requests
import os

def test_soil_analysis():
    try:
        # Server URL
        base_url = 'http://localhost:3003'
        print(f"Testing connection to {base_url}...")
        
        # First, login to get a session
        login_data = {
            'username': 'test_user',
            'password': 'test_password'
        }
        
        session = requests.Session()
        print("Attempting to login...")
        
        # Login
        login_response = session.post(f'{base_url}/login', data=login_data)
        print(f"Login response status code: {login_response.status_code}")
        print(f"Login response text: {login_response.text}")
        
        if login_response.status_code != 200:
            print("Login failed!")
            return
            
        # Now upload the soil image
        image_path = os.path.join('static', 'uploads', 'test_soil.png')
        print(f"Attempting to upload image from: {image_path}")
        print(f"Image exists: {os.path.exists(image_path)}")
        
        with open(image_path, 'rb') as f:
            files = {'soil_image': ('test_soil.png', f, 'image/png')}
            print("Sending soil analysis request...")
            response = session.post(f'{base_url}/soil-analysis', files=files)
            
        print(f"Soil analysis status code: {response.status_code}")
        print(f"Soil analysis response: {response.text}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_soil_analysis()
