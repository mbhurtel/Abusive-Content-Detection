import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.externals import joblib

classifier = joblib.load('adult_content.pkl')
cv = joblib.load('CountVectorizer')

text = input("Enter the review: ")
review = re.sub('[^a-zA-Z]', ' ', text)
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

validation_data = cv.transform(review).toarray()

prediction = classifier.predict(validation_data)

countObscene = 0
countClean = 0

for i in prediction:
    if i == 1:
        countObscene = countObscene + 1
    else:
        countClean = countClean + 1
if countObscene > countClean:
    print("Obscene")
else: 
    print("Clean")
