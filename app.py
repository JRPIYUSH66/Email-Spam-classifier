from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        email = request.form['email']
        data = vectorizer.transform([email]).toarray()
        pred = model.predict(data)[0]
        prediction = "Spam 🚨" if pred == 1 else "Not Spam ✅"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run()
