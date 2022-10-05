# library
import numpy as np
from random import randrange
import operator

"""
To evaluate the performance of the KNN algorithm
(using Euclidean distance metric),
implement a leave-one-out evaluation routine
for your algorithm. In leave-one-out validation,
we repeatedly evaluate the algorithm by removing
one data point from the training set, training the algorithm
on the remaining data set and then testing it on the point
we removed to see if the label matches or not.
Repeating this for each of the data points
gives us an estimate as to the percentage of erroneous
predictions the algorithm makes and thus a measure of the
accuracy of the algorithm for the given data. Apply your leave-one-out
validation with your KNN algorithm to the dataset for Question 1 c)
for values for K of 1, 3, 5, 7, 9, and 11 and report the results.
For which value of K do you get the best performance?
"""
# data
# Question 1 c, 1 d, 2 c, 2 d) Program Data
# replaced F=1, M=0
data = np.array(
    [
        [1.5963600450124, 75.717194178189, 23, 1],
        [1.6990610819676, 83.477307503684, 25, 0],
        [1.5052092436, 74.642420817737, 21, 1],
        [1.5738635789008, 78.562465284603, 30, 0],
        [1.796178772769, 74.566117057707, 29, 0],
        [1.6274618774347, 82.250591567161, 21, 1],
        [1.6396843250708, 71.37567170848, 20, 1],
        [1.538505823668, 77.418902097029, 32, 1],
        [1.6488692005889, 76.333044488477, 26, 1],
        [1.7233804613095, 85.812112126306, 27, 0],
        [1.7389100516771, 76.424421782215, 24, 1],
        [1.5775696242624, 77.201404139171, 29, 1],
        [1.7359417237856, 77.004988515324, 20, 0],
        [1.5510482441354, 72.950756316157, 24, 1],
        [1.5765653263667, 74.750113664457, 34, 1],
        [1.4916026885377, 65.880438515643, 28, 1],
        [1.6755053770068, 78.901754249459, 22, 0],
        [1.4805881225567, 69.652364469244, 30, 1],
        [1.6343943760912, 73.998278712613, 30, 1],
        [1.6338449829543, 79.216500811112, 27, 1],
        [1.5014451222259, 66.917339299419, 27, 1],
        [1.8575887178701, 79.942454850988, 28, 0],
        [1.6805940669394, 78.213519314007, 27, 1],
        [1.6888905106948, 83.031099742808, 20, 0],
        [1.7055120272359, 84.233282531303, 18, 0],
        [1.5681965896812, 74.753880204215, 22, 1],
        [1.6857758389206, 84.014217544019, 25, 1],
        [1.7767370337678, 75.709336556562, 27, 0],
        [1.6760125952287, 74.034126149139, 28, 0],
        [1.5999112612548, 72.040030344184, 27, 0],
        [1.6770845322305, 76.149431872551, 25, 0],
        [1.7596128136991, 87.366395298795, 29, 0],
        [1.5344541456027, 73.832214971449, 22, 1],
        [1.5992629534387, 82.4806916967, 34, 1],
        [1.6714162787917, 67.986534194515, 29, 1],
        [1.7070831676329, 78.269583353177, 25, 0],
        [1.5691295338456, 81.09431696972, 27, 0],
        [1.7767893419281, 76.910413184648, 30, 0],
        [1.5448153215763, 76.888087599642, 32, 1],
        [1.5452842691008, 69.761889289463, 30, 1],
        [1.6469991919639, 82.289126983444, 18, 1],
        [1.6353732734723, 77.829257585654, 19, 1],
        [1.7175342426502, 85.002276406574, 26, 0],
        [1.6163551692382, 77.247935733799, 21, 0],
        [1.6876845881843, 85.616829192322, 27, 0],
        [1.5472705508274, 64.474350365634, 23, 1],
        [1.558229415357, 80.382011318379, 21, 1],
        [1.6242189230632, 69.567339939973, 28, 1],
        [1.8215645865237, 78.163631826626, 22, 1],
        [1.6984142478298, 69.884030497097, 26, 0],
        [1.6468551415123, 82.666468220128, 29, 0],
        [1.5727791290292, 75.545348033094, 24, 0],
        [1.8086593470477, 78.093913654921, 27, 0],
        [1.613966988578, 76.083586505149, 23, 1],
        [1.6603990297076, 70.539053122611, 24, 0],
        [1.6737443242383, 66.042005829182, 28, 1],
        [1.6824912337281, 81.061984274536, 29, 0],
        [1.5301691510101, 77.26547501308, 22, 0],
        [1.7392340943261, 92.752488433153, 24, 0],
        [1.6427105169884, 83.322790265985, 30, 0],
        [1.5889040551166, 74.848224733663, 25, 1],
        [1.5051718284868, 80.078271153645, 31, 1],
        [1.729420786579, 81.936423109142, 26, 0],
        [1.7352568354092, 85.497712687992, 19, 0],
        [1.5056950011245, 73.726557750383, 24, 1],
        [1.772404089054, 75.534265951718, 30, 0],
        [1.5212346939173, 74.355845722315, 29, 1],
        [1.8184515409355, 85.705767969326, 25, 0],
        [1.7307897479464, 84.277029918205, 28, 1],
        [1.6372690389158, 72.289040612489, 27, 0],
        [1.6856953072545, 70.406532419182, 28, 1],
        [1.832494802635, 81.627925524191, 27, 0],
        [1.5061197864796, 85.886760677468, 31, 1],
        [1.5970906671458, 71.755566818152, 27, 1],
        [1.6780459059283, 78.900587239209, 25, 1],
        [1.6356901170146, 84.066566323977, 21, 1],
        [1.6085494116591, 70.950456539016, 30, 0],
        [1.5873479102442, 77.558144903338, 25, 0],
        [1.7542078120838, 75.3117550236, 26, 0],
        [1.642417315747, 67.97377818999, 31, 1],
        [1.5744266340913, 81.767568318602, 23, 0],
        [1.8470601407979, 68.606183538532, 30, 1],
        [1.7119387468283, 80.560922353487, 27, 1],
        [1.6169930563306, 75.538611935125, 27, 0],
        [1.6355653058986, 78.49626023408, 24, 0],
        [1.6035395957618, 79.226052358485, 33, 0],
        [1.662787957279, 76.865925681154, 25, 0],
        [1.5889291137091, 76.548543553914, 28, 1],
        [1.9058127964477, 82.56539915922, 25, 0],
        [1.694633493614, 62.870480634419, 21, 1],
        [1.7635692396034, 82.479783004684, 27, 0],
        [1.6645292231449, 75.838104636904, 29, 1],
        [1.7201968406129, 81.134689293557, 24, 1],
        [1.5775563651749, 65.920103519266, 24, 1],
        [1.6521294216004, 83.312640709417, 28, 0],
        [1.5597501915973, 76.475667826389, 30, 1],
        [1.7847561120027, 83.363676219109, 29, 0],
        [1.6765690500715, 73.98959022721, 23, 0],
        [1.6749260607992, 73.687015573315, 27, 1],
        [1.58582362825, 71.713707691505, 28, 0],
        [1.5893375739649, 74.248033504548, 27, 1],
        [1.6084440045081, 71.126430164213, 27, 1],
        [1.6048804804343, 82.049319162211, 26, 1],
        [1.5774196609804, 70.878214496062, 24, 1],
        [1.6799586185525, 75.649534976838, 29, 1],
        [1.7315642636281, 92.12183674186, 29, 0],
        [1.5563282000349, 69.312673560451, 32, 1],
        [1.7784349641893, 83.464562543, 26, 0],
        [1.7270244609765, 76.599791001341, 22, 1],
        [1.6372540837311, 74.746741127229, 30, 1],
        [1.582550559056, 73.440027907722, 23, 1],
        [1.722864383186, 79.37821152354, 20, 1],
        [1.5247544081009, 70.601290492141, 27, 1],
        [1.580858666774, 70.146982323579, 24, 1],
        [1.703343390074, 90.153276095421, 22, 1],
        [1.5339948635367, 59.675627532338, 25, 1],
        [1.8095306490733, 86.001187990639, 20, 0],
        [1.7454786971676, 85.212429336602, 22, 0],
        [1.6343303342105, 85.46378358014, 32, 0],
        [1.5983479173071, 79.323905480504, 27, 1],
    ]
)


# classifier
class kNNClassifier:
    def __init__(self, k=3, distanceMetric="euclidean"):
        pass

    def fit(self, xTrain, yTrain):

        assert len(xTrain) == len(yTrain)
        self.trainData = xTrain
        self.trainLabels = yTrain

    def getNeighbors(self, testRow):

        calcDM = distanceMetrics()
        distances = []
        for i, trainRow in enumerate(self.trainData):
            if self.distanceMetric == "euclidean":
                distances.append(
                    [
                        trainRow,
                        calcDM.euclideanDistance(testRow, trainRow),
                        self.trainLabels[i],
                    ]
                )
            elif self.distanceMetric == "manhattan":
                distances.append(
                    [
                        trainRow,
                        calcDM.manhattanDistance(testRow, trainRow),
                        self.trainLabels[i],
                    ]
                )
            elif self.distanceMetric == "hamming":
                distances.append(
                    [
                        trainRow,
                        calcDM.hammingDistance(testRow, trainRow),
                        self.trainLabels[i],
                    ]
                )
            distances.sort(key=operator.itemgetter(1))

        neighbors = []
        for index in range(self.k):
            neighbors.append(distances[index])
        return neighbors

    def predict(self, xTest, k, distanceMetric):

        self.testData = xTest
        self.k = k
        self.distanceMetric = distanceMetric
        predictions = []

        for i, testCase in enumerate(self.testData):
            neighbors = self.getNeighbors(testCase)
            output = [row[-1] for row in neighbors]
            prediction = max(set(output), key=output.count)
            predictions.append(prediction)

        return predictions


class distanceMetrics:
    def euclideanDistance(self, vector1, vector2):

        self.vectorA, self.vectorB = vector1, vector2
        if len(self.vectorA) != len(self.vectorB):
            raise ValueError("Undefined for sequences of unequal length.")
        distance = 0.0
        for i in range(len(self.vectorA) - 1):
            distance += (self.vectorA[i] - self.vectorB[i]) ** 2
        return (distance) ** 0.5


def printMetrics(actual, predictions):

    assert len(actual) == len(predictions)
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predictions[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# leave one out: similar to kfold with kfolds equal to samples in df
class kFoldCV:
    def __init__(self):
        pass

    def crossValSplit(self, dataset, numFolds):

        dataSplit = list()
        dataCopy = list(dataset)
        foldSize = int(len(dataset) / numFolds)
        for _ in range(numFolds):
            fold = list()
            while len(fold) < foldSize:
                index = randrange(len(dataCopy))
                fold.append(dataCopy.pop(index))
            dataSplit.append(fold)
        return dataSplit

    def kFCVEvaluate(self, dataset, numFolds, *args):

        knn = kNNClassifier()
        folds = self.crossValSplit(dataset, numFolds)
        scores = list()
        for fold in folds:
            trainSet = list(folds)
            trainSet.remove(fold)
            trainSet = sum(trainSet, [])
            testSet = list()
            for row in fold:
                rowCopy = list(row)
                testSet.append(rowCopy)

            trainLabels = [row[-1] for row in trainSet]
            trainSet = [train[:-1] for train in trainSet]
            knn.fit(trainSet, trainLabels)

            actual = [row[-1] for row in testSet]
            testSet = [test[:-1] for test in testSet]

            predicted = knn.predict(testSet, *args)

            accuracy = printMetrics(actual, predicted)
            scores.append(accuracy)

        print("Maximum Accuracy: %3f%%" % max(scores))
        print("Mean Accuracy: %.3f%%" % (sum(scores) / float(len(scores))))


# implementation with dataset minus age col

kfcv = kFoldCV()

# data prep
trainFeatures = []
for row in data[:, :-1]:
    index = row[0:]
    temp = [item for item in index]
    trainFeatures.append(temp)

# 1
print("k=1")
kfcv.kFCVEvaluate(trainFeatures, data.shape[0], 1, 'euclidean')
print("----------------------------------------------")

# 3
print("k=3")
kfcv.kFCVEvaluate(trainFeatures, data.shape[0], 3, 'euclidean')
print("----------------------------------------------")

# 5
print("k=5")
kfcv.kFCVEvaluate(trainFeatures, data.shape[0], 5, 'euclidean')
print("----------------------------------------------")

# 7
print("k=7")
kfcv.kFCVEvaluate(trainFeatures, data.shape[0], 7, 'euclidean')
print("----------------------------------------------")

# 9
print("k=9")
kfcv.kFCVEvaluate(trainFeatures, data.shape[0], 9, 'euclidean')
print("----------------------------------------------")

# 11
print("k=11")
kfcv.kFCVEvaluate(trainFeatures, data.shape[0], 11, 'euclidean')
print("----------------------------------------------")


# conclusion
"""
The model accuracy drops significantly with the highest
accuracy when modeling without the age data being 12.5% with k == 11.
This implies that age is significant to the model when predicting gender.
A second observation is the fact that the highest value of k==1
has the highest accuracy unlike in the previous case where
smaller values of k (==3 or ==5) revealed the highest accuracy.
This could indicate the higher the number of features,
the better chance of getting a high accuracy when the number of
k-neighbors is small and vice versa.
"""
