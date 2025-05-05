# 🧠 Spam Message Classifier (ML + Flask)

This is a simple spam classifier built with Python, Flask, and Scikit-learn. It uses a Naive Bayes machine learning model to detect whether a given text message is **Spam** or **Not Spam**.

---

## 🚀 Features

- Train a spam detection model on SMS data
- Use NLP and sklearn for preprocessing and classification
- Web interface using Flask to enter messages and get predictions

---

## 🛠️ Tech Stack

- Python 3
- Flask
- Pandas
- Scikit-learn
- NLTK

---

## 📂 Project Structure
spam_classifier_project/
│
├── app.py # Flask app
├── model.pkl # Saved trained ML model
├── vectorizer.pkl # Saved TfidfVectorizer
├── spam.csv # Dataset (SMS messages)
├── templates/
│ └── index.html # Frontend form
├── README.md # Project overview
└── .gitignore


---

## 🔧 How to Run

1. Install dependencies:
   ```bash
   pip install flask pandas scikit-learn nltk



2.Run the app:

python app.py

3.Open your browser and go to:

http://127.0.0.1:5000/


