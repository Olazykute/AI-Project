from sklearn.tree import DecisionTreeClassifier
import Projet_IA as P

# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv  
# now you can import normally from model_selection
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV

print("Decision Tree Model")
X, y = P.data_transfer(P.data_0.filtered)

X_train, X_test, y_train, y_test = P.train_test_split(X, y, test_size=0.4)
X_train = P.data_scaling(X_train)
X_test = P.data_scaling(X_test)

model_clf,  prediction_test, prediction_train = P.training_model(DecisionTreeClassifier(), X_train, y_train, X_test, y_test)

# P.save_model(model_clf, 'Vehicle_prediction_DecisionTree')
P.Model_Report(model_clf, X_test, y_test, prediction_test)
P.disp_confusionMatrix(model_clf, y_test, prediction_test, 'Confusion matrix for DecisionTreeClassifier model')


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