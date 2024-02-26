import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split

data=pl.read_csv("sensor_raw.csv")

X = data[:,1:len(data)] #les caractéristiques
y = data[:, 0]  #les résulats (classes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
modele = MultinomialNB()
modele.fit(X_train, y_train)

y_pred = modele.predict(X_test)
from sklearn.metrics import accuracy_score
print("precsion : ", accuracy_score(y_test, y_pred))
