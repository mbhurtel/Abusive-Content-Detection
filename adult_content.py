# Natural Language Processing

# Importing the libraries
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('bad-words_new.csv', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 6880):
    review = re.sub('[^a-zA-Z]', ' ', dataset['content'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#print(y_pred)

#print(classifier)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

text = "i loved it"
review1 = re.sub('[^a-zA-Z]', ' ', text)
review1 = review1.lower()
review1 = review1.split()
ps = PorterStemmer()
review1 = [ps.stem(word) for word in review1 if not word in set(stopwords.words('english'))]

z = cv.transform(review1).toarray()
result = classifier.predict(z)


    
from sklearn.externals import joblib
joblib.dump(classifier, 'adult_content.pkl') 
joblib.dump(cv, 'CountVectorizer')
#classifier = joblib.load('review.pkl')