#importing library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#create a bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#Splitting the dataset into training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Fitting Naive Bayes to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifier.predict(X_test)

#test_text = "really loved this movie"
#review1 = re.sub('[^a-zA-Z]', ' ', test_text)
#review1 = review1.lower()
#review1 = review1.split()
#ps1 = PorterStemmer()
#review1 = [ps1.stem(word) for word in review1 if not word in stopwords.words('english')]
#review1 = ' '.join(review1)
#matrix = [review1]
#
#print(review1)
#
#valid = cv.fit_transform(matrix).toarray()
#
#classifier.predict(valid)


#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)