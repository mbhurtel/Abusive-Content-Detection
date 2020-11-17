import re
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.externals import joblib
#from sklearn.feature_extraction.text import CountVectorizer

classifier = joblib.load('review.pkl')
cv = joblib.load('CountVectorizer')

text = "Restaurant was awful"
review = re.sub('[^a-zA-Z]', ' ', text)
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

validation_data = cv.transform(review).toarray()

prediction = classifier.predict(validation_data)

countLiked = 0
countUnliked = 0

for i in prediction:
    if i == 1:
        countLiked = countLiked + 1
    else:
        countUnliked = countUnliked + 1
if countLiked > countUnliked:
    print("Positive response")
else: 
    print("Negative response")
