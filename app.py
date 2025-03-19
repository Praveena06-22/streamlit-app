import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from soil_classifier import SoilClassifier
from config import Config
from train_models import predict_crop

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY

# Initialize soil classifier
soil_classifier = SoilClassifier()
soil_classifier.train(Config.SOIL_TYPE_DATASET)

# Test credentials (in a real app, use a database)
USERS = {
    'test_user': 'test_password'
}

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in USERS and USERS[username] == password:
            session['logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/soil-analysis', methods=['GET', 'POST'])
def soil_analysis():
    if not session.get('logged_in'):
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        if 'soil_image' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
            
        file = request.files['soil_image']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
            
        if file and Config.allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Analyze soil
                result = soil_classifier.predict(filepath)
                
                # Get soil type characteristics from config
                soil_type_key = result['soil_type'].lower().replace(' ', '_')
                soil_info = Config.SOIL_TYPES.get(soil_type_key, {
                    'properties': {},
                    'suitable_crops': []
                })
                
                return render_template('soil_result.html',
                    image_filename=filename,
                    soil_type=result['soil_type'],
                    confidence=result['confidence'],
                    soil_properties=soil_info['properties'],
                    suitable_crops=soil_info['suitable_crops']
                )
            except Exception as e:
                flash(f'Error analyzing soil: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a PNG or JPG image.', 'error')
            return redirect(request.url)
                
    return render_template('soil_analysis.html')

@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if not session.get('logged_in'):
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            # Get form data and convert to the format expected by predict_crop
            features = {
                'N': float(request.form.get('N', request.form.get('nitrogen', 0))),
                'P': float(request.form.get('P', request.form.get('phosphorus', 0))),
                'K': float(request.form.get('K', request.form.get('potassium', 0))),
                'pH': float(request.form.get('pH', request.form.get('ph', 0))),
                'temperature': float(request.form['temperature']),
                'rainfall': float(request.form['rainfall'])
            }
            
            # Validate parameters
            Config.validate_parameters(features)
            
            # Get crop recommendations
            recommended_crops = predict_crop(features, Config.N_RECOMMENDATIONS)
            
            return render_template('recommendation_result.html',
                                nitrogen=features['N'],
                                phosphorus=features['P'],
                                potassium=features['K'],
                                ph=features['pH'],
                                temperature=features['temperature'],
                                rainfall=features['rainfall'],
                                recommended_crops=recommended_crops)
        except ValueError as e:
            flash(f'Invalid input: {str(e)}', 'error')
            return redirect(url_for('crop_recommendation'))
        except Exception as e:
            flash(f'Error processing request: {str(e)}', 'error')
            return redirect(url_for('crop_recommendation'))
    
    return render_template('crop_recommendation.html')

if __name__ == '__main__':
    app.run(debug=True, port=3004, host='0.0.0.0')
