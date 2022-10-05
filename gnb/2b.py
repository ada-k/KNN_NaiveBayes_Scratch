import numpy as np
from math import sqrt, pi, exp


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
Implement the Gaussian Na ̈ ıve Bayes Classifier for this problem
"""


# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [
        (mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del summaries[-1]
    return summaries


# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev**2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[
            class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(
                row[i], mean, stdev
                )
    return probabilities


# Predict the class for a given row
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    return predictions


# predict using x_test
dataset = np.column_stack((x_train, y_train))
preds = naive_bayes(dataset, x_test)
print(["F" if x == 1 else "M" for x in preds])
