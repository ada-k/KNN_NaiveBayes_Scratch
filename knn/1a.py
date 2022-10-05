# library
import numpy as np


# Question 1 a) and 2 a) Training Data:
x_train = np.array(
    [
        [1.6530190426733, 72.871146648479, 24],
        [1.6471384909498, 72.612785314988, 34],
        [1.6472055785348, 73.53968351051, 33],
        [1.7323008914951, 76.067870338779, 30],
        [1.6750702657911, 81.05582111533, 30],
        [1.5780970716644, 64.926084680188, 30],
        [1.6587629355524, 69.38092449041, 30],
        [1.6763295980234, 77.062295990149, 31],
        [1.7187224085504, 62.112923317057, 37],
        [1.5202218226439, 66.151444019603, 27],
        [1.5552689261884, 66.076386143769, 31],
        [1.6969333189258, 77.45386244568, 34],
        [1.6887980792886, 76.489640732464, 37],
        [1.5213552893624, 63.952944947832, 35],
    ],
    dtype=float,
)
x_train = x_train.astype(float)

y_train = np.array(
    ["W", "W", "M", "M", "M", "W", "M", "M", "W", "W", "W", "M", "M", "W"]
)
# label encode: 0s:M, 1s:W
y_train = np.array(
    ["1", "1", "0", "0", "0", "1", "0", "0", "1", "1", "1", "0", "0", "1"]
)
y_train = y_train.astype(int)

# Question 1 a) and 2 a) Test Data:
x_test = np.array(
    [
        [1.62065758929, 59.376557437583, 32],
        [1.7793983848363, 72.071775670801, 36],
        [1.7004576585974, 66.267508112786, 31],
        [1.6591086215159, 61.751621901787, 29],
    ],
    dtype=float,
)
x_test = x_test.astype(float)


"""
## K Nearest Neighbor

1. Consider the problem where we want to predict the gender of a person
from a set of input parameters,namely height, weight, and age.
a) Using Cartesian distance, Manhattan distance and Minkowski distance of
order 3 as the similarity measurements show the results of the gender
prediction for the Evaluation data that is listed below generated
training data for values of K of 1, 3, and 7. Include the intermediate
steps (i.e., distance calculation, neighbor selection, and prediction).

"""


class KNearestNeighbors:
    def __init__(
        self,
        X_train,
        y_train,
        neighbors=1,
        weights="uniform",
        distance="cartesian"
    ):

        self.X_train = X_train
        self.y_train = y_train

        self.neighbors = neighbors
        self.weights = weights

        self.n_classes = 2

        self.distance = distance

    # distance metrics
    def cartesian_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    def manhattan_distance(self, a, b):
        return sum(abs(val1 - val2) for val1, val2 in zip(a, b))

    def minkowski_distance(self, a, b, p=3):
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
                enum_neigh, key=lambda x: x[1])[: self.neighbors]

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


# cartesian distance
print("cartesian distance: 1 neighbor")
knn = KNearestNeighbors(x_train, y_train, neighbors=1, distance="cartesian")
pred = knn.predict(x_test)
print(["F" if x == 1 else "M" for x in pred])
print("-------------------------------")


print("cartesian distance: 3 neighbors")
knn = KNearestNeighbors(x_train, y_train, neighbors=3, distance="cartesian")
pred = knn.predict(x_test)
print(["F" if x == 1 else "M" for x in pred])
print("-------------------------------")


print("cartesian distance: 7 neighbors")
knn = KNearestNeighbors(x_train, y_train, neighbors=7, distance="cartesian")
pred = knn.predict(x_test)
print(["F" if x == 1 else "M" for x in pred])
print("-------------------------------")


# manhattan distance
print("manhattan distance: 1 neighbor")
knn = KNearestNeighbors(x_train, y_train, neighbors=1, distance="manhattan")
pred = knn.predict(x_test)
print(["F" if x == 1 else "M" for x in pred])
print("-------------------------------")


print("manhattan distance: 3 neighbors")
knn = KNearestNeighbors(x_train, y_train, neighbors=3, distance="manhattan")
pred = knn.predict(x_test)
print(["F" if x == 1 else "M" for x in pred])
print("-------------------------------")


print("manhattan distance: 7 neighbors")
knn = KNearestNeighbors(x_train, y_train, neighbors=7, distance="manhattan")
pred = knn.predict(x_test)
print(["F" if x == 1 else "M" for x in pred])
print("-------------------------------")


# minkowski distance
print("minkowski distance: 1 neighbor")
knn = KNearestNeighbors(x_train, y_train, neighbors=1, distance="minkowski")
pred = knn.predict(x_test)
print(["F" if x == 1 else "M" for x in pred])
print("-------------------------------")


print("minkowski distance: 3 neighbors")
knn = KNearestNeighbors(x_train, y_train, neighbors=3, distance="minkowski")
pred = knn.predict(x_test)
print(["F" if x == 1 else "M" for x in pred])
print("-------------------------------")


print("minkowski distance: 7 neighbors")
knn = KNearestNeighbors(x_train, y_train, neighbors=7, distance="minkowski")
pred = knn.predict(x_test)
print(["F" if x == 1 else "M" for x in pred])
print("-------------------------------")
