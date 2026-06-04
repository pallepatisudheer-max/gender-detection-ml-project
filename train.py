from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd
import pickle

data = pd.read_csv('dataset.csv')

model = make_pipeline(
    CountVectorizer(analyzer='char', ngram_range=(2,3)),
    MultinomialNB()
)

model.fit(data['name'], data['gender'])

with open('gender_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Model trained and saved as gender_model.pkl')
