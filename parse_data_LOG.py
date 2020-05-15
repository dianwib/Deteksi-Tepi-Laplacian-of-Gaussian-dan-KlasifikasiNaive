import cv2
import os
import matplotlib.pyplot as plt
import numpy as np



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

# print(file_data_sehat.shape)
# print(file_data_sakit.shape)

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
    filename_1 = 'HASIL/Original image.jpg'
    filename_2 = 'HASIL/Gray image.jpg'

    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite(filename_1, image)
    cv2.imwrite(filename_2, gray)

    #buat LOG_1
    gray_1 = 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray_1=np.array(gray_1,dtype='uint8')

    # print(gray_1,"gray")
    # print(gray.shape)
    # a = Image.fromarray(gray_1)
    # a.show()
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return gray_1

#murni LOG input data file==url image
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
    filename_1 = 'HASIL/LOG_asli.jpg'
    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite(filename_1, abs_dst)
    return np.mean(abs_dst),np.std(abs_dst)

#LOG dgn komposisi greuscale input data file==image greyscale (grey_1)
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
    filename_1 = 'HASIL/LOG_edit.jpg'
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


# test=file_ata_sakit[19]
# a=convert_rgb_to_greyscale(test)
# LOG(test)
# LOG_1(a)
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