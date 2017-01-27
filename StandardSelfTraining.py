from sklearn import neighbors
from sklearn import svm
from sklearn import tree


class StandardSelfTraining:
    @staticmethod
    def KNN():
        """
        Create Standard Self-Training classifier with NN base classifier"""
        base_clf = neighbors.KNeighborsClassifier(
            n_neighbors=3,
            metric="euclidean",
            n_jobs=2  # Parallelize work on CPUs
        )
        return StandardSelfTraining("Self-Training (KNN)", base_clf)

    @staticmethod
    def SMO():
        """
        Create Standard Self-Training classifier with SVM base classifier
        the SVM classifier has been trained using SMO algorithm
        """
        base_clf = svm.SVC(
            C=1.0,
            kernel='poly',
            degree=1,
            tol=0.001,
            # Epsilon parameter missing?
        )
        return StandardSelfTraining("Self-Training (SVM)", base_clf)

    @staticmethod
    def CART():
        base_clf = tree.DecisionTreeClassifier(
            criterion='entropy',
            #splitter='best',
            #max_depth=None,
            #min_samples_split=2,
            min_samples_leaf=2,
            #min_weight_fraction_leaf=0.0,
            #max_features=None,
            #random_state=None,
            #max_leaf_nodes=None,
            #min_impurity_split=1e-07,
            #class_weight=None,
            #presort=False,
        )
        return StandardSelfTraining("Self-Training (CART)", base_clf)

    def __init__(self, name, base_classifier, max_iterations=40):
        self.name = name
        self.base_classifier = base_classifier
        self.max_iterations = max_iterations

    def __str__(self):
        return "Classifier: " + self.name + "\nParameters: " + str(self.base_classifier.get_params())

    def fit(self, X, y):
        stable = False
        iteration = 0
        # Iterate until the result is stable or max_iterations is reached
        while not stable and (iteration < self.max_iterations):
            new_y = self._fit_iteration(X, y)
            # Check if the result has stabilised
            stable = (y == new_y).all()
            y = new_y
            iteration += 1

    def _fit_iteration(self, X, y):
        clf = self.base_classifier
        # Fit a classifier on already labeled data
        labeled = y != "unlabeled"
        clf.fit(X[labeled], y[labeled])
        # Predict on all the training data
        return clf.predict(X)

    def predict(self, X):
        return self.base_classifier.predict(X)

    def score(self, X, y):
        return self.base_classifier.score(X, y)