from sklearn.tree import DecisionTreeClassifier
import Projet_IA as P
from sklearn.tree import plot_tree
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

model_clf,  prediction_test, prediction_train = P.training_model(DecisionTreeClassifier(
    criterion="entropy",
    max_depth=30, min_samples_split=2,
    min_samples_leaf=1, max_features="sqrt", max_leaf_nodes=None, min_impurity_decrease=0.1),
    X_train, y_train, X_test, y_test)

# P.save_model(model_clf, 'Vehicle_prediction_DecisionTree')
P.Model_Report(model_clf, X_test, y_test, prediction_test)
P.disp_confusionMatrix(model_clf, y_test, prediction_test,
                       'Confusion matrix for DecisionTreeClassifier model')

""" 
param_grid = {  # This part is to research the best parameters to maximize the model's accuracy
    'criterion': ['gini', 'entropy'], #entropy
    'max_depth': [None, 10, 20, 30], #30
    'min_samples_split': [2, 5, 10],#2
    'min_samples_leaf': [1, 2, 4], #1
    'max_features': ['sqrt', 'log2'],#sqrt
    'max_leaf_nodes': [None, 10, 20, 30],#None
    'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3],#0.1
}
HGSearch = HalvingGridSearchCV(
    model_clf, param_grid, cv=5, factor=2, max_resources=100)
HGSearch.fit(X, y)

print("Best parameters found: ", HGSearch.best_params_)
print("Best score: ", HGSearch.best_score_)
'''
# df = pl.DataFrame(HGSearch.cv_results_)
"""


# Plot the Decision Tree
P.plt.figure(figsize=(20,12))
plot_tree(model_clf, feature_names = P.data_0.filtered[1, 1:len(P.data_0.filtered)].columns ,class_names=['Sudden Acceleration', "Sudden Right Turn", 'Sudden Left Turn', 'Sudden Break'],filled=True);
P.plt.title ('Decision Tree')
P.plt.show()
'''
# Plot learning curve
X = P.np.concatenate((X_train, X_test), axis=0)
y = P.np.concatenate((y_train, y_test), axis=0)
P.plot_learning_curve(DecisionTreeClassifier(
    criterion="entropy",
    max_depth=30, min_samples_split=2,
    min_samples_leaf=1, max_features="sqrt", max_leaf_nodes=None, min_impurity_decrease=0.1),
    'Learning Curve For Decision Tree Model', X, y, cv=5)
'''