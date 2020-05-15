import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from math import exp
from math import pi
from random import seed
from random import randrange
import scipy
# image = cv2.imread('dataset-daun-jagung-master/bercak-daun/0a403456-5c5e-4aad-aa89-a118175c6ddd___RS_GLSp 4501_final_masked.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# cv2.imshow('Original image', image)
# cv2.imshow('Gray image', gray)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

folder_citra_sehat='dataset-daun-jagung-master/sehat'
folder_citra_sakit='dataset-daun-jagung-master/sakit'


def parse_dataset_rgb(folder):
    data_temp=[]
    for filename in os.listdir(folder):

        data_temp.append(folder+'/'+filename)
    data_temp=np.array(data_temp)
    return data_temp

# data_sehat=parse_dataset_rgb(folder_citra_sehat)
# data_sakit=parse_dataset_rgb(folder_citra_sakit)
# np.save('list_file_data_sehat.npy', data_sehat)
# np.save('list_file_data_sakit.npy', data_sakit)
file_data_sehat=np.load('list_file_data_sehat.npy',allow_pickle=True)
file_data_sakit=np.load('list_file_data_sakit.npy',allow_pickle=True)

print(file_data_sehat.shape)
print(file_data_sakit.shape)

def convert_rgb_to_greyscale(file):
    image = cv2.imread(file)
    # print(image.dtype)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]
    # print (R,"R")
    # print(0.2989 * R, "*R")
    # print(G, "G")
    # print( 0.5870 * G, "*G")
    # print(B, "B")
    # print(0.1140*B, "*B")

    # cv2.imshow('Original image', image)
    # cv2.imshow('Gray image', gray)
    # Filename
    filename_1 = 'HASIL/Original1.jpg'
    filename_2 = 'HASIL/Gray_asli1.jpg'
    filename_3 = 'HASIL/Gray_edit1.jpg'

    # Using cv2.imwrite() method
    # Saving the image


    gray_1 = 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray_1=np.array(gray_1,dtype='uint8')

    cv2.imwrite(filename_1, image)
    cv2.imwrite(filename_2, gray)
    cv2.imwrite(filename_3, gray_1)

    # print(gray_1,"gray")
    # print(gray.shape)
    # a = Image.fromarray(gray_1)
    # a.show()
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return gray_1

def LOG(file):

    # [variables]
    # Declare the variables we are going to use
    ddepth = cv2.CV_16S
    kernel_size = 3
    window_name = "Laplace Demo"
    # [variables]
    # [load]
    imageName = file
    src = cv2.imread(cv2.samples.findFile(imageName), cv2.IMREAD_COLOR) # Load an image
    # print(src.dtype)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image')
        print ('Program Arguments: [image_name -- default lena.jpg]')
        return -1
    # [load]
    # [reduce_noise]
    # Remove noise by blurring with a Gaussian filter
    src = cv2.GaussianBlur(src, (3, 3), 0)
    # print(src.shape)
    # [reduce_noise]
    # [convert_to_gray]
    # Convert the image to grayscale
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # print(src_gray.dtype)


    # [convert_to_gray]
    # Create Window
    # cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    # [laplacian]
    # Apply Laplace function
    dst = cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)
    # [laplacian]
    # [convert]
    # converting back to uint8
    abs_dst = cv2.convertScaleAbs(dst)
    print(np.mean(abs_dst), "mean")
    print(np.std(abs_dst), "std")
    # [convert]
    # [display]
    # cv2.imshow(window_name, abs_dst)
    # cv2.waitKey(0)
    # [display]
    # return 0
    # Filename
    filename_1 = 'HASIL/LOG_asli1.jpg'
    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite(filename_1, abs_dst)
    return np.mean(abs_dst),np.std(abs_dst)


def LOG_1(file):
    ddepth = cv2.CV_16S
    kernel_size = 3
    window_name = "Laplace Demo_1"
    # cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    # [laplacian]
    # Apply Laplace function
    dst = cv2.Laplacian(file, ddepth, ksize=kernel_size)
    # [laplacian]
    # [convert]
    # converting back to uint8
    abs_dst = cv2.convertScaleAbs(dst)
    print(np.mean(abs_dst),"mean")
    print(np.std(abs_dst),"std")

    # [convert]
    # [display]
    # cv2.imshow(window_name, abs_dst)
    # cv2.waitKey(0)
    # [display]
    # return 0

    # Filename
    filename_1 = 'HASIL/LOG_edit1.jpg'
    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite(filename_1, abs_dst)
    return np.mean(abs_dst), np.std(abs_dst)


def plot_histogram(folder='HASIL'):
    i=1
    for filename in os.listdir(folder):
        print(filename)
        img = cv2.imread(folder+'/'+filename)
        # cv2.imshow('Original image', img)

        # find frequency of pixels in range 0-255
        histr = cv2.calcHist([img], [0], None, [256], [0, 256])

        # show the plotting graph of an image
        plt.figure(i)
        plt.plot(histr)
        plt.title(filename)
        i+=1

    # plt.show()


test=file_data_sakit[22]
a=convert_rgb_to_greyscale(test)
LOG(test)
LOG_1(a)
# plot_histogram()
def parse_data(data,tipe_kelas):
    temp_list = []
    for i in range(len(data)):
        test = data[i]
        a = convert_rgb_to_greyscale(test)
        rata_1, std_1 = LOG(test)
        rata_2, std_2 = LOG_1(a)

        temp = [rata_1, std_1, rata_2, std_2, tipe_kelas]
        temp_list.append(temp)
    return temp_list

# list_data_sakit=parse_data(file_data_sakit,"Sakit")
# list_data_sehat=parse_data(file_data_sehat,"Sehat")
# np.save('list_data_sehat.npy', list_data_sehat)
# np.save('list_data_sakit.npy', list_data_sakit)




data_sehat=np.load('list_data_sehat.npy',allow_pickle=True)
data_sakit=np.load('list_data_sakit.npy',allow_pickle=True)

print(data_sehat.shape)
print(data_sakit.shape)
#grey ori mean,grey ori std,grey no ori mean,grey no ori std,class

class K_mean(object):
    def __init__(self,data_sehat=data_sehat,data_sakit=data_sakit):
        dataset=self.join_dataset(data_sehat=data_sehat,data_sakit=data_sakit)
        # convert class column to integers
        self.dict_kelas=self.str_column_to_int(dataset, len(dataset[0]) - 1)
        self.dataset = np.array(dataset, dtype=float).tolist()

    def join_dataset(self,data_sehat=data_sehat,data_sakit=data_sakit):
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
            print('[%s] => %d' % (value, i))

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
        print(probabilities)
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


Kmean=K_mean()
dataset=Kmean.dataset
# print(dataset[0])



# print(dataset)

# evaluate algorithm
summaries = Kmean.summarize_by_class(dataset)
print(summaries)
probabilities = Kmean.calculate_class_probabilities(summaries, dataset[0])
print(probabilities)
# evaluate algorithm
n_folds = 5
scores = Kmean.evaluate_algorithm(dataset, Kmean.naive_bayes, n_folds)
print('Scores: ', scores)
print('Mean Accuracy: ' , (sum(scores)/float(len(scores))))

##PREDICT

# fit model
model = Kmean.summarize_by_class(dataset)
# define a new record

test = file_data_sakit[21]
a = convert_rgb_to_greyscale(test)
rata_1, std_1 = LOG(test)
rata_2, std_2 = LOG_1(a)
row =[rata_1,std_1,rata_2,std_2]
# predict the label
label = Kmean.predict(model, row)

print('Data= ',row,' Predicted: ' , label,'/',list(Kmean.dict_kelas.keys())[list(Kmean.dict_kelas.values()).index(label)])




'''




NEXT
1. cari nilai mean
2. cari nilai stardart deviasi pad masing" kelas
3. improve pake metode klasifikasi
Berdasarkan hasil penelitian, didapatkan beberapa kesimpulan sebagai berikut.
 Pola daun sehat berupa garis lurus sejajar, tanpa distorsi pola lain.
 Berdasarkan analisis histogram, pola daun yang sehat memiliki komposisi yang relatif stabil antara warna merah, hijau, dan biru, dibandingkan dengan daun yang terindikasi penyakit.
 Berdasarkan hasil perbandingan nilai intensitas warna, semakin parah penyakit, maka nilai intensitas warna juga semakin rendah. Artinya ketika daun terindikasi penyakit yang lebih parah, warna akan berubah menjadi kusam.
'''