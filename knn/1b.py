# library
import numpy as np

"""
Implement the KNN algorithm for this problem.
Your implementation should work with different
training data sets as well as different
values of K and allow to input a data point for the prediction.
"""

class KNearestNeighbors:
    def __init__(
        self,
        X_train,
        y_train,
        n_neighbors=1,
        weights="uniform",
        distance="cartesian"
    ):

        self.X_train = X_train
        self.y_train = y_train

        self.n_neighbors = n_neighbors
        self.weights = weights

        self.n_classes = 2

        self.distance = distance

    # distance metrics
    def cartesian_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    def manhattan_distance(self, a, b):
        return sum(abs(val1 - val2) for val1, val2 in zip(a, b))

    def minkowski_distance(a, b, p=3):
        return np.sum(np.abs(a - b) ** p, axis=1) ** (1 / p)

    # neighnor selection
    def kneighbors(self, X_test, return_distance=False):

        dist = []
        neigh_ind = []

        if self.distance == "cartesian":
            point_dist = [
                self.cartesian_distance(
                    x_test, self.X_train) for x_test in X_test
            ]

        elif self.distance == "manhattan":
            point_dist = [
                self.manhattan_distance(
                    x_test, self.X_train) for x_test in X_test
            ]

        elif self.distance == "minkowski":
            point_dist = [
                self.minkowski_distance(
                    x_test, self.X_train) for x_test in X_test
            ]

        for row in point_dist:
            enum_neigh = enumerate(row)
            sorted_neigh = sorted(
                enum_neigh, key=lambda x: x[1])[: self.n_neighbors]

            ind_list = [tup[0] for tup in sorted_neigh]
            dist_list = [tup[1] for tup in sorted_neigh]

            dist.append(dist_list)
            neigh_ind.append(ind_list)

        if return_distance:
            return np.array(dist), np.array(neigh_ind)

        return np.array(neigh_ind)

    # prediction
    def predict(self, X_test):

        if self.weights == "uniform":
            neighbors = self.kneighbors(X_test)
            y_pred = np.array(
                [
                    np.argmax(np.bincount(self.y_train[neighbor]))
                    for neighbor in neighbors
                ]
            )

            return y_pred

        if self.weights == "distance":

            dist, neigh_ind = self.kneighbors(X_test, return_distance=True)

            inv_dist = 1 / dist

            mean_inv_dist = inv_dist / np.sum(inv_dist, axis=1)[:, np.newaxis]

            proba = []

            for i, row in enumerate(mean_inv_dist):

                row_pred = self.y_train[neigh_ind[i]]

                for k in range(self.n_classes):
                    indices = np.where(row_pred == k)
                    prob_ind = np.sum(row[indices])
                    proba.append(np.array(prob_ind))

            predict_proba = np.array(proba).reshape(
                X_test.shape[0], self.n_classes)

            y_pred = np.array([np.argmax(item) for item in predict_proba])

            return y_pred

    # accuracy score
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)

        return float(sum(y_pred == y_test)) / float(len(y_test))


# sample prediction w/k==6
test = np.array([
    [1.5963600450124, 75.717194178189, 23],
    [1.6990610819676, 83.477307503684, 25],
    [1.5052092436, 74.642420817737, 21],
    [1.5738635789008, 78.562465284603, 30],
    [1.796178772769, 74.566117057707, 29]
])

train = np.array([
    [1.6876845881843, 85.616829192322, 27],
    [1.5472705508274, 64.474350365634, 23],
    [1.558229415357, 80.382011318379, 21],
    [1.6242189230632, 69.567339939973, 28],
    [1.8215645865237, 78.163631826626, 22],
    [1.6984142478298, 69.884030497097, 26],
    [1.6468551415123, 82.666468220128, 29],
    [1.5727791290292, 75.545348033094, 24],
    [1.8086593470477, 78.093913654921, 27],
    [1.613966988578, 76.083586505149, 23],
    [1.6603990297076, 70.539053122611, 24],
    [1.6737443242383, 66.042005829182, 28],
    [1.6824912337281, 81.061984274536, 29],
    [1.5301691510101, 77.26547501308, 22],
])
labels = np.array([
    1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0
])

print("cartesian distance: 6 neighbors")
knn = KNearestNeighbors(train, labels, n_neighbors=6, distance="cartesian")
pred = knn.predict(test)
print(["F" if x == 1 else "M" for x in pred])
print("-------------------------------")