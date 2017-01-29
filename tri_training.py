import sklearn
from math import floor
from sklearn.metrics import accuracy_score
import numpy as np


class TriTraining:
    def __init__(self, name, base_classifier):
        self.name = name
        self.base_classifier = base_classifier

    def __str__(self):
        return "Classifier: " + self.name + "\nParameters: " + str(self.base_classifier.get_params())

    def fit(self, X, y):
        X = X.as_matrix()
        y = y.as_matrix()
        unlabeled = y == "unlabeled"
        labeled = ~unlabeled

        #Initialize
        self.thirds = [Third(sklearn.base.clone(self.base_classifier), X[labeled], y[labeled]) for i in range(3)]
        third_rotations =  [rotate(self.thirds, i) for (i,_) in enumerate(self.thirds)]

        changed = True
        while changed:
            changed = False
            for t1, t2, t3 in third_rotations:
                changed |= t1.train(t2, t3, X[unlabeled])


    def predict(self, X):
        predictions = np.asarray([third.predict(X) for third in self.thirds])
        #print("Predictions", predictions)
        #maj = np.asarray([np.argmax(np.bincount(predictions[:, c])) for c in range(predictions.shape[1])])
        #return maj
        import scipy
        return scipy.stats.mstats.mode(predictions).mode[0]


    def score(self, X, y_true):
        y_true = y_true.astype(int)
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)

class Third:
    def __init__(self, classifier, labeled_X, labeled_y):
        self.classifier = classifier
        self.labeled_X = labeled_X
        self.labeled_y = labeled_y

        sample = sklearn.utils.resample(self.labeled_X, self.labeled_y)  # BootstrapSample(L)
        self.classifier.fit(*sample)  # Learn(Si)
        self.err_prime = 0.5  # e'i = 0.5
        self.l_prime = 0.0  # l'i = 0.0

    def update(self, L_X, L_y, error):
        X = np.append(self.labeled_X, L_X, axis=0)
        y = np.append(self.labeled_y, L_y, axis=0)
        self.classifier.fit(X, y)
        self.err_prime = error
        self.l_prime = len(L_X)

    def train(self, t1, t2, unlabeled_X):
        L_X = []
        L_y = []
        error = self.measure_error(t1, t2)
        if (error >= self.err_prime):
            return False

        for X in unlabeled_X:
            X = X.reshape(1, -1)
            y = t1.predict(X)
            if y == t2.predict(X):
                L_X.append(X)
                L_y.append(y)

        count_of_added = len(L_X)
        # Turn the python list of chosen samples into numpy array
        L_X = np.concatenate(L_X)
        L_y = np.concatenate(L_y)

        if (self.l_prime == 0):
            self.l_prime = floor(error / (self.err_prime - error) + 1)

        if self.l_prime >= count_of_added:
            return False
        if error * count_of_added < self.err_prime * self.l_prime:
            self.update(L_X, L_y, error)
            return True
        if self.l_prime > error / (self.err_prime - error):
            n = floor(self.err_prime * self.l_prime / error - 1)
            L_X, L_y = sklearn.utils.resample(L_X, L_y, replace=False, n_samples=n)

            self.update(L_X, L_y, error)
            return True
        return False

    def measure_error(self, third_1, third_2):
        prediction_1 = third_1.predict(self.labeled_X)
        prediction_2 = third_2.predict(self.labeled_X)
        both_incorrect = np.count_nonzero((prediction_1 != self.labeled_y) & (prediction_2 != self.labeled_y))
        both_same = np.count_nonzero(prediction_1 == prediction_2)
        error = both_incorrect/both_same
        return error

    def predict(self, *args, **kwargs):
        return self.classifier.predict(*args, **kwargs)


#Helper for rotating a list
def rotate(l, n):
    return l[n:] + l[:n]


if __name__ == "__main__":
    import base_classifiers
    path_to_datasets = "../Datasets/"

    # All datasets used for testing
    dataset_names = ["bupa", "abalone"]
    labeling_rates = [10, 20, 30, 40]


    def load_dataset(path):
        """Load one dataset"""
        return pd.read_csv(path, header=None, sep=", ", engine="python", comment="@")


    def load_datasets(dataset_name, labeling_rate=10):
        """ Load 3 datasets: training, transitive and testing"""
        partial_path = "{0}SSC_{1}labeled/{2}/{2}-10-1".format(path_to_datasets, labeling_rate, dataset_name)
        dataframes = {t: load_dataset(partial_path + t + ".dat") for t in ["tra", "trs", "tst"]}
        return dataframes

    TriTraining("test", base_classifiers.knn)
