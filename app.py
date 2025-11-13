# app.py
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('fake_review_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    data = [review]
    vect = vectorizer.transform(data).toarray()
    prediction = model.predict(vect)
    result = "Fake Review ❌" if prediction[0] == 1 else "Genuine Review ✅"
    return render_template('index.html', prediction=result, review=review)

if __name__ == '__main__':
    app.run(debug=True)
