# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Example dataset (you can replace with real dataset)
data = {
    'review': [
        'This product is great and works perfectly',
        'Worst product ever, do not buy!',
        'Amazing quality! I recommend this to everyone',
        'Buy this product now! Best deal ever!!!',
        'Completely fake and useless item',
        'Really loved it, totally worth it',
        'Good product, it worth'
    ],
    'label': [0, 1, 0, 1, 1, 0]  # 0 = Genuine, 1 = Fake
}

df = pd.DataFrame(data)

# Preprocess and split data
X = df['review']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open('fake_review_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("✅ Model trained and saved successfully!")
