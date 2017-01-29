from sklearn import neighbors
from sklearn import svm
from sklearn import tree

knn = neighbors.KNeighborsClassifier(
    n_neighbors=3,
    metric="euclidean",
    #n_jobs=2  # Parallelize work on CPUs
)

smo = svm.SVC(
        C=1.0,
        kernel='poly',
        degree=1,
        tol=0.001,
        # Epsilon parameter missing?
    )


cart = tree.DecisionTreeClassifier(
    criterion='entropy',
    # splitter='best',
    # max_depth=None,
    # min_samples_split=2,
    min_samples_leaf=2,
    # min_weight_fraction_leaf=0.0,
    # max_features=None,
    # random_state=None,
    # max_leaf_nodes=None,
    # min_impurity_split=1e-07,
    # class_weight=None,
    # presort=False,
)