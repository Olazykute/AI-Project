import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.svm import SVC
from joblib import dump, load
from Projet_IA import data_0
from GaussianNB_Model_Training import data_transfer, data_scaling, save_model, load_model, training_model, plot_learning_curve
# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv  
# now you can import normally from model_selection
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV

print("Decision Tree Model")
X, y = data_transfer(data_0.filtered)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
X_train = data_scaling(X_train)
X_test = data_scaling(X_test)

model_clf = training_model(DecisionTreeClassifier(), X_train, y_train, X_test, y_test)

save_model(model_clf, 'Vehicle_prediction_DecisionTree')

param_grid = { # This part is to research the best parameters to maximize the model's accuracy
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10,20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4], 
    'max_features': ['sqrt', 'log2'], 
    'max_leaf_nodes': [None, 10, 20, 30],
    'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3],
}
HGSearch = HalvingGridSearchCV(model_clf, param_grid, cv=5, factor=2, max_resources=100)
HGSearch.fit(X, y)

print("Best parameters found: ", HGSearch.best_params_)
print("Best score: ", HGSearch.best_score_)

# df = pl.DataFrame(HGSearch.cv_results_)

'''
# Plot learning curve
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)
plot_learning_curve(DecisionTreeClassifier(), 'Learning Curve For Decision Tree Model', X, y, cv=5)
'''