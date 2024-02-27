import csv
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from nltk.corpus import stopwords
import webbrowser
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB

def learnAndCreateClassifer( data ):
	X = data[ 'Problem' ]
	y = data[ 'Solution ID' ]
	u = data[ 'Solution label' ]
	z = data[ 'Confluence link' ]

	with open( 'learning.data', 'wb' ) as dataFile:
	    pickle.dump( data, dataFile )

	solution_id_link_mapping = {}
	solution_id_label_mapping = {}
	
	for count in range( len( y ) ):
		solution_id_link_mapping[ y[ count ] ] = z[ count ]
		solution_id_label_mapping[ y[ count ] ] = u[ count ]
	problems = []

	stemmer = WordNetLemmatizer()

	for sen in range(0, len(X)):
	    # Remove all the special characters
	    problem = re.sub(r'\W', ' ', str(X[sen]))

	    # remove all single characters
	    problem = re.sub(r'\s+[a-zA-Z]\s+', ' ', problem)

	    # Remove single characters from the start
	    problem = re.sub(r'\^[a-zA-Z]\s+', ' ', problem)

	    # Substituting multiple spaces with single space
	    problem = re.sub(r'\s+', ' ', problem, flags=re.I)

	    # Removing prefixed 'b'
	    problem = re.sub(r'^b\s+', '', problem)

	    # Converting to Lowercase
	    problem = problem.lower()

	    # Lemmatization
	    problem = problem.split()

	    problem = [stemmer.lemmatize(word) for word in problem]
	    problem = ' '.join(problem)

	    problems.append(problem)

	vectorizer = CountVectorizer( min_df=5, max_df=0.7, stop_words=stopwords.words('english') )
	X = vectorizer.fit_transform(problems).toarray()

	tfidfconverter = TfidfTransformer()
	X = tfidfconverter.fit_transform(X).toarray()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

	classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
	classifier.fit(X_train, y_train)

	y_pred = classifier.predict(X_test)

	print( confusion_matrix( y_test,y_pred ) )
	print( classification_report( y_test,y_pred) )
	print( accuracy_score( y_test, y_pred ) )

	with open('bugs_classifier', 'wb') as picklefile:
	    pickle.dump(classifier,picklefile)

	with open('covnersionvector', 'wb') as vectorfile:
	    pickle.dump( vectorizer, vectorfile )

	with open('tfidconverter', 'wb') as tfidfile:
	    pickle.dump( tfidfconverter, tfidfile )

	with open('idtolabelmapping', 'wb') as idToLabelMapper:
	    pickle.dump( solution_id_label_mapping, idToLabelMapper )

	with open('idtolinkmapping', 'wb') as idToLinkMapping:
	    pickle.dump( solution_id_link_mapping, idToLinkMapping )
if __name__ == "__main__":
	print "First time learning"
	csv_file = "hackathon_model_data.csv"
	data = pd.read_csv( csv_file )
	learnAndCreateClassifer( data )

def feedBackLoop( data, catagoryID, catagory, link = None ):
	print data, catagoryID, catagory
	storedData = pickle.load( open( 'learning.data', 'rb' ) )
	df = pd.DataFrame( { 'Problem': [ data ],
			'Solution ID': [ catagoryID ],
			'Solution label': [ catagory ],
			'Confluence link': [ link ] } )
	updatedData = storedData.append( df, ignore_index = True, sort=False )
	learnAndCreateClassifer( updatedData )
	updatedData.to_csv('hackathon_model_data.csv', index=False)
