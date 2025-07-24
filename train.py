import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = [
    "Congratulations! You've won a prize",
    "Please send your account details",
    "Project meeting today",
    "Can we have a call tomorrow?",
    "URGENT: Your account is compromised"
]
labels = [1, 1, 0, 0, 1]  # 1 = spam, 0 = not spam

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
