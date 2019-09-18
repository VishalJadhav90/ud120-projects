#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image

def accuracy(results, labels_test):
    from sklearn.metrics import accuracy_score
    print(accuracy_score(results, labels_test))

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(features_train, labels_train)
results = clf_knn.predict(features_test)
accuracy(results, labels_test)

from sklearn.ensemble import RandomForestClassifier
clf_for = RandomForestClassifier(criterion="entropy", n_estimators=5)
clf_for.fit(features_train, labels_train)
results = clf_for.predict(features_test)
accuracy(results, labels_test)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
clf_ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion="entropy", max_depth=1), n_estimators=40)
clf_ada.fit(features_train, labels_train)
results = clf_ada.predict(features_test)
accuracy(results, labels_test)


clf = clf_ada
try:
    prettyPicture(clf, features_test, labels_test)
    output_image("test.png", "png", open("test.png", "rb").read())
except NameError:
    pass
