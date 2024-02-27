import numpy as np
import re
import nltk
from sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier
#nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords

movie_data = load_files("./txt_sentoken")

X, y = movie_data.data, movie_data.target

#print( X )
#print ( y )

documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

#with open('text_classifier', 'wb') as picklefile:
#    pickle.dump(classifier,picklefile)

#with open('covnersionvector', 'wb') as vectorfile:
#    pickle.dump( vectorizer, vectorfile )

#with open('tfidconverter', 'wb') as tfidfile:
#    pickle.dump( tfidfconverter, tfidfile )

solution = { 0 : "Solution_geolocation", 1: "Solution_rtsp", 2: "Solution_sim_switch" }
print( "label" )
text = "SIM switching not working with Huwaei"
text = vectorizer.transform([text]).toarray()
text = tfidfconverter.transform(text).toarray()
label = classifier.predict(text)[0]
print( label )
print( solution[ int( label ) ] )

