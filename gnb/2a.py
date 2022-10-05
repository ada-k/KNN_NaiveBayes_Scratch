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
Using the data from Problem 2, build a Gaussian Na ̈ ıve Bayes
classifier for this problem. For this you have to:
learn Gaussian distribution parameters for each input
data feature, i.e. for p(height|W ), p(height|M ),
p(weight|W ),p(weight|M ), p(age|W ), p(age|M ).
"""


def mean_var(X, y):
    n_features = X.shape[1]
    m = np.ones((2, n_features))
    v = np.ones((2, n_features))
    n_0 = np.bincount(y)[np.nonzero(np.bincount(y))[0]][0]
    x0 = np.ones((n_0, n_features))
    x1 = np.ones((X.shape[0] - n_0, n_features))

    k = 0
    for i in range(0, X.shape[0]):
        if y[i] == 0:
            x0[k] = X[i]
            k = k + 1
    k = 0
    for i in range(0, X.shape[0]):
        if y[i] == 1:
            x1[k] = X[i]
            k = k + 1

    for j in range(0, n_features):
        m[0][j] = np.mean(x0.T[j])
        v[0][j] = np.var(x0.T[j]) * (n_0 / (n_0 - 1))
        m[1][j] = np.mean(x1.T[j])
        v[1][j] = np.var(x1.T[j]) * (
            (X.shape[0] - n_0) / ((X.shape[0] - n_0) - 1))
    return m, v


(means, variance) = mean_var(x_train, y_train)
print(f"Means|W: \nHeight: {means[1][0]}\nWeight: {means[1][1]}\nAge: {means[1][2]}\n")
print(f"Means|M: \nHeight:{means[0][0]}\nWeight: {means[0][1]}\nAge: {means[0][2]}\n")
print("\n")
print(
    f"Variance|W: \nHeight: {variance[1][0]}\nWeight: {variance[1][1]}\nAge: {variance[1][2]}\n"
)
print(
    f"Variance|M: \nHeight: {variance[0][0]}\nWeight: {variance[0][1]}\nAge: {variance[0][2]}\n"
)
