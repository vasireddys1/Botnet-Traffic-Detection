import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import h5py
import os

from sklearn import model_selection, feature_selection, utils, ensemble, linear_model, metrics
from sklearn.tree import export_graphviz

print("Import data")

X = pd.read_hdf('data_window_botnet3.h5', key='data')
X.reset_index(drop=True, inplace=True)

X2 = pd.read_hdf('data_window3_botnet3.h5', key='data')
X2.reset_index(drop=True, inplace=True)

X = X.join(X2)

X.drop('window_id', axis=1, inplace=True)

y = X['Label_<lambda>']
X.drop('Label_<lambda>', axis=1, inplace=True)

labels = np.load("data_window_botnet3_labels.npy",allow_pickle=True)

print(X.columns.values)
print(labels)
print(np.where(labels == 'flow=From-Botne')[0][0])

y_bin6 = y==np.where(labels == 'flow=From-Botne')[0][0]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_bin6, test_size=0.33, random_state=123456)

print(X_train.shape)
X_train_new, y_train_new = utils.resample(X_train, y_train, n_samples=X_train.shape[0]*20, random_state=123456)

print("y", np.unique(y, return_counts=True))
print("y_train", np.unique(y_train_new, return_counts=True))
print("y_test", np.unique(y_test, return_counts=True))

print("Random Forest Classifier")

clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=123456, verbose=1, class_weight=None)
clf.fit(X_train_new, y_train_new)
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train_new.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train_new.shape[1]), indices)
plt.xlim([-1, X_train_new.shape[1]])
plt.savefig("random_forest_feature_importance.jpg", format="jpg",bbox_inches='tight')
plt.show()

for tree_in_forest in clf.estimators_:
    export_graphviz(tree_in_forest,out_file='tree.dot',
    feature_names=X.columns,
    filled=True,
    rounded=True)
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    name = 'tree' + str(i_tree)
    graph.write_png(name+  '.png')
    os.system('dot -Tpng tree.dot -o tree.png')
    i_tree +=1
print("Train")
y_pred_train = clf.predict(X_train_new)
print("accuracy score = ", metrics.balanced_accuracy_score(y_train_new, y_pred_train))
precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_train_new, y_pred_train)
print("precision = ", precision[1])
print("recall = ", recall[1])
print("fbeta_score = ", fbeta_score[1])
print("support = ", support[1])

print("Test")
y_pred_test = clf.predict(X_test)
print("accuracy score = ", metrics.balanced_accuracy_score(y_test, y_pred_test))
precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_test, y_pred_test)
print("precision = ", precision[1])
print("recall = ", recall[1])
print("fbeta_score = ", fbeta_score[1])
print("support = ", support[1])
