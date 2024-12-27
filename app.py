from flask import Flask, render_template, request, jsonify
import pickle
from scipy.sparse import hstack
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
with open('saved_models/hybrid_model.pkl', 'rb') as model_file:
    saved_objects = pickle.load(model_file)

model = saved_objects['model']
vectorizer = saved_objects['vectorizer']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        title = request.form.get('title', '').strip()
        company_profile = request.form.get('company_profile', '').strip()
        description = request.form.get('description', '').strip()
        requirements = request.form.get('requirements', '').strip()
        benefits = request.form.get('benefits', '').strip()
        telecommuting = request.form.get('telecommuting', '').strip()
        has_company_logo = request.form.get('has_company_logo', '').strip()
        has_questions = request.form.get('has_questions', '').strip()

        # Check if text fields are empty
        if not (title or company_profile or description or requirements or benefits):
            return jsonify({'error': 'Text fields are required. Please fill out at least one.'})

        # Validate and convert numerical inputs
        try:
            telecommuting = int(telecommuting) if telecommuting else 0
            has_company_logo = int(has_company_logo) if has_company_logo else 0
            has_questions = int(has_questions) if has_questions else 0
        except ValueError:
            return jsonify({'error': 'Numerical fields must be valid integers (0 or 1).'})

        # Ensure numerical inputs are binary
        if not all(val in [0, 1] for val in [telecommuting, has_company_logo, has_questions]):
            return jsonify({'error': 'Numerical fields must be binary (0 or 1).'})

        # Combine text features
        text_features = ' '.join(filter(None, [title, company_profile, description, requirements, benefits]))
        text_vectorized = vectorizer.transform([text_features])

        # Combine with numerical features
        numerical_features = np.array([[telecommuting, has_company_logo, has_questions]])
        combined_features = hstack([text_vectorized, numerical_features])

        # Make prediction
        prediction = model.predict(combined_features)[0]
        result = "Fake" if prediction == 1 else "Real"

        return jsonify({'result': result})
    except Exception as e:
        # General error handling
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
