import joblib
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# Load model
model = joblib.load("final_models/model.pkl")
vectorizer = joblib.load("final_models/vectorizer.pkl")

# Test input
text = input("Enter text: ")

text = clean_text(text)
text_vec = vectorizer.transform([text])

prediction = model.predict(text_vec)

print("Predicted Topic:", prediction[0])