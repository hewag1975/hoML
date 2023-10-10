import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

## classification
iris = load_iris()
iris.keys()
iris['target_names']

X = iris.data[:, 2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

from sklearn.tree import export_graphviz
export_graphviz(
    tree_clf,
    out_file="/home/hendrik/Downloads/iris_tree.dot",
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

## predict class probability
tree_clf.predict_proba([[5, 1.5]])
tree_clf.predict([[5, 1.5]])


## regression
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X_quad = np.random.rand(200, 1) - 0.5  # a single random input feature
y_quad = X_quad ** 2 + 0.025 * np.random.randn(200, 1)

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_quad, y_quad)

export_graphviz(
    tree_reg,
    out_file="/home/hendrik/Downloads/regression_tree.dot",
    feature_names=["x1"],
    rounded=True,
    filled=True
)

from graphviz import Source
tst = Source.from_file("/home/hendrik/Downloads/regression_tree.dot")
tst.view()


## exercise 7 (tune decision tree)
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=10000, noise=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

cls = DecisionTreeClassifier()

prm = {
    "max_depth": range(2, 10)#[2, 4, 6],
    "max_leaf_nodes": range(2, 12)# [3, 6, 9]
}

grd = GridSearchCV(cls, param_grid=prm, cv=3)
grd.fit(X_train, y=y_train)
grd.best_params_['max_depth']

cls_opt = DecisionTreeClassifier(
    max_depth=grd.best_params_['max_depth'], 
    max_leaf_nodes=grd.best_params_['max_leaf_nodes']
)

## alternative training with best params
cls_opt = DecisionTreeClassifier()
cls_opt.set_params(**grd.best_params_)

cls_opt.fit(X_train, y=y_train)
prd = cls_opt.predict(X_test)

accuracy_score(y_test, y_pred=prd)


## exercise 8 (grow forest)
from sklearn.model_selection import ShuffleSplit

rs = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
rs.get_n_splits(X)

for i, (train_index, test_index) in enumerate(rs.split(X)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
