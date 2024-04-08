from sklearn.ensemble import RandomForestClassifier
import Projet_IA as P
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.tree import plot_tree

# KNN, Decision tree/RandomForest, SVM,, Optimisation Bayesienne to see what are the best parameters to tune our model
print("Random Forest Model")
X, y = P.data_transfer(P.data_0.filtered)

X_train, X_test, y_train, y_test = P.train_test_split(X, y, test_size=0.4)
X_train = P.data_scaling(X_train)
X_test = P.data_scaling(X_test)
X = P.data_scaling(X)

'''
model_clf, prediction_test, prediction_train = P.training_model(RandomForestClassifier(
    n_estimators=100, criterion='entropy',
    max_depth=10, min_samples_split=5,
    min_samples_leaf=2, max_features='log2',
    max_leaf_nodes=10, bootstrap=True),
    X_train, y_train, X_test, y_test)
'''
model_clf, prediction_test, prediction_train = P.training_model(RandomForestClassifier(),X_train, y_train, X_test, y_test)
# P.save_model(model_clf, 'Vehicle_prediction_RandomForest')
P.Model_Report(model_clf, X, y, y_test, prediction_test)
P.disp_confusionMatrix(model_clf, y_test, prediction_test, 'Confusion matrix for RandomForestClassifier model')

tree_to_plot = model_clf.estimators_[5]
P.plt.figure(figsize=(16,10))
plot_tree(tree_to_plot, feature_names = P.data_0.filtered[1, 1:len(P.data_0.filtered)].columns ,class_names=[
    'Sudden Acceleration', 'Sudden Right Turn', 'Sudden Left Turn', 'Sudden Break'],filled=True)
P.plt.title ('RandomForest tree number 5')
tree2_to_plot = model_clf.estimators_[7]
P.plt.figure(figsize=(14,12))
plot_tree(tree2_to_plot, feature_names = P.data_0.filtered[1, 1:len(P.data_0.filtered)].columns ,class_names=[
    'Sudden Acceleration', 'Sudden Right Turn', 'Sudden Left Turn', 'Sudden Break'],filled=True)
P.plt.title ('RandomForest tree number 7')
P.plt.show()

'''
# This part is to research the best parameters to maximize the model's accuracy
# After running this part, we found the best parameters to be the ones we used in the model
param_grid = {
    'n_estimators': [100, 200, 300], #100
    'criterion': ['gini', 'entropy'], #entropy
    'max_depth': [None, 10, 20], #10
    'min_samples_split': [2, 5, 10], #5
    'min_samples_leaf': [1, 2, 4], #2
    'max_features': ['sqrt', 'log2'],#log2
    'max_leaf_nodes': [None, 10, 20], # 10
    'bootstrap': [True, False] #True
}

HGSearch = HalvingGridSearchCV(model_clf, cv=5, factor=2, max_resources=100, param_grid=param_grid)
HGSearch.fit(X_train, y_train)

print("Best parameters found: ", HGSearch.best_params_)
print("Best score: ", HGSearch.best_score_)
'''
param_distributions = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

search = HalvingRandomSearchCV(model_clf, param_distributions, cv=5, factor=2, max_resources=100)
search.fit(X_train, y_train)
print("Best Params:" , search.best_params_)
print("Best score: ", search.best_score_)

'''
# Plot learning curve
X = P.np.concatenate((X_train, X_test), axis=0)
y = P.np.concatenate((y_train, y_test), axis=0)
P.plot_learning_curve(model_clf,'Learning Curve For Random Forest Model', X, y, cv=5)
'''
