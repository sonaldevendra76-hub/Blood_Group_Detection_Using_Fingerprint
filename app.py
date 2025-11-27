from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from keras.models import load_model

# Load the ML model
model = load_model("model/model.h5")
main_model = load_model("model/model.h5")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = 'your_secret_key'

# Create the upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file uploaded!')
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file!')
        return redirect(url_for('home'))

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Validate and predict blood group
        is_valid = is_valid_fingerprint(file_path)
        if is_valid:
            blood_group = get_blood_group(file_path)
            return render_template('result.html', image_url=url_for('static', filename=f'uploads/{filename}'), blood_group=blood_group)
        else:
            flash('Invalid fingerprint image. Please upload a valid fingerprint.')
            return redirect(url_for('home'))

    flash('Something went wrong. Please try again.')
    return redirect(url_for('home'))

def is_valid_fingerprint(image_path):
    """Checks if the uploaded image is a valid fingerprint."""
    return model_predict_fc(image_path, model) != 0

def get_blood_group(image_path):
    """Predicts the blood group using the main model."""
    return predict_image(main_model, image_path)

if __name__ == '__main__':
    app.run(debug=True)
