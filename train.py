import nltk
nltk.download('stopwords')

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter = '\t', quoting = 3)
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

corpus=[]

for i in range(0, 900):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

cv = CountVectorizer(max_features = 1420)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Create the train folder if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Saving BoW dictionary to later use in prediction
bow_path = 'models/c1_BoW_Sentiment_Model.pkl'
pickle.dump(cv, open(bow_path, "wb"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Saving the classifier model to later use in prediction
joblib.dump(classifier, 'models/c2_Classifier_Sentiment_Model') 



# Model performance measure
# y_pred = classifier.predict(X_test)

# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# accuracy_score(y_test, y_pred)