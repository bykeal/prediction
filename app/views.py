from django.shortcuts import render,redirect
import sys
from subprocess import run,PIPE
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
# Create your views here.
def index(request):
	return render(request,'ugochi.html')

def external(request):
	text1 = request.POST.get('CSC_311')
	text2 = request.POST.get('CSC_221')
	text3 = request.POST.get('CSC_211')
	index = predict(text1,text2,text3)
	data = index[0][0]
	return render(request, 'output.html',{'data':data,'text1':text1,'text2':text2,'text3':text3})

def predict(text1,text2,text3):
	index = []
	
	csv = "C://Users//chukwunenyea//Desktop//predict//predict//templates//ugochi.csv"
	names = ['Digital_Design_CSC211','Assembly_language_CSC221', 'Micoprocessor_CSC311', 'Computer_architecture_CSC311']
	data = pd.read_csv(csv, names=names)
	X = data.iloc[:, :-1].values
	X = X[1:]
	y = data.iloc[:, 3].values
	y = y[1:]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
	#naive bayes
	classifier2 = MultinomialNB()
	classifier2.fit(X_train, y_train)
	y_predict2 = classifier2.predict(X_test)
	#KNN
	scalar = StandardScaler()
	scalar.fit(X_train)
	X_trained = scalar.transform(X_train)
	classifier = KNeighborsClassifier(n_neighbors=78)
	classifier.fit(X_train, y_train)
	y_predict = classifier.predict(X_test)

	result = StandardScaler()
	result.fit([[text1,text2,text3]])
	resultant = result.transform([[text1,text2,text3]])
	output = classifier.predict(resultant)
	output2 = classifier2.predict([[text1,text2,text3]]) 

	if output > output2:
	    index.append(output)
	else:
	    index.append(output2)
	if output2 == output:
		index.append(output2)
	else:
		pass
	return index

