import numpy as np
from math import sqrt
from math import exp
from math import pi
from random import randrange
from parse_data_LOG import data_sakit,data_sehat

class K_mean(object):
    def __init__(self,data_sehat=data_sehat,data_sakit=data_sakit):
        dataset=self.join_dataset(data_sehat,data_sakit)
        # convert class column to integers
        self.dict_kelas=self.str_column_to_int(dataset, len(dataset[0]) - 1)
        self.dataset = np.array(dataset, dtype=float).tolist()

    def join_dataset(self,data_sehat,data_sakit):
        temp=[]
        def parse(data):
            for i in range(len(data)):
                temp.append(data[i])

        parse(data_sehat)
        parse(data_sakit)
        return np.array(temp)


    # Convert string column to integer
    def str_column_to_int(self,dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        '''
           ubah kelas menjadi intger
           [Sehat] => 0
           [Sakit] => 1
           '''
        for i, value in enumerate(unique):
            lookup[value] = i
            # print('[%s] => %d' % (value, i))

        for row in dataset:
            row[column] = lookup[row[column]]

        return lookup

    # Split a dataset into k folds
    def cross_validation_split(self,dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for _ in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    # Calculate accuracy percentage
    def accuracy_metric(self,actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self,dataset, algorithm, n_folds, *args):
        folds = self.cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores

    # Split the dataset by class values, returns a dictionary
    def separate_by_class(self,dataset):
        separated = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if (class_value not in separated):
                separated[class_value] = list()
            separated[class_value].append(vector)
        return separated

    # Calculate the mean of a list of numbers
    def mean(self,numbers):
        return sum(numbers) / float(len(numbers))

    # Calculate the standard deviation of a list of numbers
    def stdev(self,numbers):
        avg = self.mean(numbers)
        variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
        return sqrt(variance)

    # Calculate the mean, stdev and count for each column in a dataset
    def summarize_dataset(self,dataset):
        summaries = [(self.mean(column), self.stdev(column), len(column)) for column in zip(*dataset)]
        del (summaries[-1])
        return summaries

    # Split dataset by class then calculate statistics for each row
    def summarize_by_class(self,dataset):
        separated = self.separate_by_class(dataset)
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = self.summarize_dataset(rows)
        return summaries

    # Calculate the Gaussian probability distribution function for x
    def calculate_probability(self,x, mean, stdev):
        exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent

    # Calculate the probabilities of predicting each class for a given row
    def calculate_class_probabilities(self,summaries, row):
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[class_value] *= self.calculate_probability(row[i], mean, stdev)
        return probabilities

    # Predict the class for a given row
    def predict(self,summaries, row):
        probabilities = self.calculate_class_probabilities(summaries, row)
        # print(probabilities)
        self.probabilities=probabilities
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    # Naive Bayes Algorithm
    def naive_bayes(self,train, test):
        summarize = self.summarize_by_class(train)
        predictions = list()
        for row in test:
            output = self.predict(summarize, row)
            predictions.append(output)
        return (predictions)