from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
import string
nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)

# Load and preprocess dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

df['message'] = df['message'].apply(clean_text)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

def predict_message(msg):
    msg_clean = clean_text(msg)
    msg_vect = vectorizer.transform([msg_clean])
    pred = model.predict(msg_vect)[0]
    return "Spam" if pred == 1 else "Not Spam"

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        msg = request.form["msg"]
        result = predict_message(msg)
    return render_template_string("""
    <form method='POST'>
        <input name='msg' placeholder='Enter a message'>
        <button type='submit'>Check</button>
    </form>
    <h2>{{ result }}</h2>
    """, result=result)

if __name__ == "__main__":
    app.run(debug=True)
