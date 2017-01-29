class StandardSelfTraining:
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