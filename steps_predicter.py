import sys
import pickle
import webbrowser
import argparse
import pandas as pd
from termcolor import colored
from csv_reader_problem_classifier import feedBackLoop

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--bugDescription", nargs='?', const='Geo location not working', type=str, help = "Enter bug description")
args = vars(ap.parse_args())
if len( sys.argv ) < 2:
	ap.print_help( sys.stderr )
	sys.exit( 1 )

bugDescription = args[ 'bugDescription' ]
vectorizer = pickle.load( open( 'covnersionvector', 'rb' ) )
tfidfconverter = pickle.load( open( 'tfidconverter', 'rb' ) )
idToLabelMapper = pickle.load( open( 'idtolabelmapping', 'rb' ) )
idToLinkMapper = pickle.load( open( 'idtolinkmapping', 'rb' ) )
model = pickle.load( open( 'bugs_classifier','rb') )
bugDescription = vectorizer.transform( [ bugDescription ] ).toarray()
bugDescription = tfidfconverter.transform( bugDescription ).toarray()
label = model.predict( bugDescription )[ 0 ]
#webbrowser.open( lableToLinkMapper[ label ] )
#print( model.predict_proba( bugDescription ) )
print( "catagory: " , idToLabelMapper[ label ] )
isCorrect = raw_input( "Is it correct?: " )
if isCorrect not in [ "yes", "y" ]:
	for label, link in idToLabelMapper.items():
		print label, link
	catagoryID, catagory, link = raw_input( colored("Enter correct id, catagory, and link for debugging: ", "red" ) ).split( ', ' )
	feedBackLoop( args[ 'bugDescription' ], int( catagoryID ), catagory, link )
else:
	redirectToConfluencePage = raw_input( colored("Do you want to redirect to relevant information page? ", "green" ))
	if redirectToConfluencePage in [ "yes", "y", 'Y', 'YES' ]:
		webbrowser.open( idToLinkMapper[ label ] )
