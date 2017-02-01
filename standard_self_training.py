import numpy as np
class StandardSelfTraining:
    def __init__(self, name, base_classifier, max_iterations=40):
        self.name = name
        self.base_classifier = base_classifier
        self.max_iterations = max_iterations

    def __str__(self):
        return "Classifier: " + self.name + "\nParameters: " + str(self.base_classifier.get_params())

    def fit(self, X, y):
        y = np.copy(y)#copy in order not to change original data
        
        all_labeled = False
        iteration = 0
        # Iterate until the result is stable or max_iterations is reached
        while not all_labeled  and (iteration < self.max_iterations):
            self._fit_iteration(X, y)
            all_labeled = (y != "unlabeled").all()
            iteration += 1
        print(iteration,end="")

    def _fit_iteration(self, X, y):
        threshold = 0.7
        
        clf = self.base_classifier
        # Fit a classifier on already labeled data
        labeled = y != "unlabeled"
        clf.fit(X[labeled], y[labeled])

        probabilities = clf.predict_proba(X)
        threshold = min(threshold, probabilities[~labeled].max()) #Get at least the best one
        over_thresh = probabilities.max(axis=1)>=threshold
        
        y[~labeled & over_thresh] = clf.predict(X[~labeled & over_thresh])

    def predict(self, X):
        return self.base_classifier.predict(X)

    def score(self, X, y):
        return self.base_classifier.score(X, y)