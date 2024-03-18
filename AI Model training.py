import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.svm import SVC

data=pd.read_csv("Features_by_window_size/sero_features_4.csv")

X = data.iloc[:, 1:len(data)] #les caractéristiques
y = data.iloc[:, 0]  #les résulats (classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modele = GaussianNB()
modele.fit(X_train, y_train)

y_pred = modele.predict(X_test)

print("precsion : ", accuracy_score(y_test, y_pred))

ValidationCurveDisplay.from_estimator(
   SVC(kernel="linear"), X, y, param_name="C", param_range=np.logspace(-7, 3, 10)
)
plt.show()