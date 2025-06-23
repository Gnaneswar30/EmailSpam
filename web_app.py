from flask import Flask, request, render_template_string
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# Load HTML template
with open("index.html", "r", encoding="utf-8") as file:
    html_template = file.read()

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        message = request.form["message"]
        cleaned = clean_text(message)
        vect_msg = vectorizer.transform([cleaned])
        pred = model.predict(vect_msg)[0]
        result = "Spam" if pred == 1 else "Not Spam"
    return render_template_string(html_template, result=result)

if __name__ == "__main__":
    app.run(debug=True)