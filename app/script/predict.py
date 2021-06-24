import numpy as np
import pandas as pd
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB

query1 = sys.argv[]
query2 = sys.argv[1]
query3 = sys.argv[2]

csv = "./ugochi.csv"
names = ['Digital_Design_CSC211','Assembly_language_CSC221', 'Micoprocessor_CSC311', 'Computer_architecture_CSC311']
data = pd.read_csv(csv, names=names)
print(data.head())
print(data.shape)

X = data.iloc[:, :-1].values
X = X[1:]
y = data.iloc[:, 3].values
y = y[1:] 
# print(Y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

classifier2 = MultinomialNB()
classifier2.fit(X_train, y_train)

scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=78)
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)

# print(X_train)
# print('...................................')
# print(y_predict)
result = StandardScaler()
result.fit([[query1,query2,query3]])
resultant = result.transform([[query1,query2,query3]])

output = classifier.predict(resultant)
return(output)
# correctpred = (y_test == classifier.predict(X_test)).sum()
# correctpred2 = (y_test == classifier2.predict(X_test)).sum()

# print(int(query1) + int(query2) + int(query3))