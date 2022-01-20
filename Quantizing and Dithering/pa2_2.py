import sys
import cv2
import numpy as np
import pa1_2 as FS

def colorTransfer(source,target):

    sourceImg,sRGB,RGB1,source_img=initialize(source)
    targetImg, tRGB, RGB2,target_img = initialize(target)

    source_img = np.hstack((RGB1[0], RGB1[1],RGB1[2]))
    target_img = np.hstack((RGB2[0], RGB2[1],RGB2[2]))

    rgb2lms = np.array([[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]])

    logLabMatrix1 = np.array([[1 / np.sqrt(3), 0, 0], [0, 1 / np.sqrt(6), 0], [0, 0, 1 / np.sqrt(2)]])
    logLabMatrix2 = np.array([[1, 1, 1], [1, 1, -2], [1, -1, 0]])



    # to LMS space
    s_lms = np.dot(rgb2lms, source_img.transpose())
    t_lms = np.dot(rgb2lms, target_img.transpose())

    # log 10
    sourceLMS,loggedSource = zeros2one(s_lms)
    targetLMS,loggedTarget=zeros2one(t_lms)


    # to LAB space
    p1 = np.dot(logLabMatrix1, logLabMatrix2)

    sourceLab = np.dot(p1, loggedSource)
    targetLab = np.dot(p1, loggedTarget)

    # to statistics
    sourceMean,sourceStd=staticalAlignment(p1,sourceLab)
    targetMean,targetStd=staticalAlignment(p1,targetLab)

    standartF =np.divide(targetStd,sourceStd)

    # apply the statistical alignment for shape
    res_lab = np.zeros(sourceLab.shape)

    for i in range(0, 3):
        res_lab[i, :] = (sourceLab[i, :] - sourceMean[i]) * standartF[i] + targetMean[i]

    bmatrix2 = np.array([[np.sqrt(3) / 3, 0, 0], [0, np.sqrt(6) / 6, 0], [0, 0, np.sqrt(2) / 2]])
    cmatrix2 = np.array([[1, 1, 1], [1, 1, -1], [1, -2, 0]])

    # convert resource back to LMS shape
    LMS_res = np.dot(np.dot(cmatrix2, bmatrix2), res_lab)

    for idx in range(0, 3):
        LMS_res[idx, :] = np.power(10, LMS_res[idx, :])

    # convert back to RGB
    lms2rgb = np.array([[4.4679, -3.5873, 0.1193], [-1.2186, 2.3809, -0.1624], [0.0497, -0.2439, 1.2045]])
    est_im = np.dot(lms2rgb, LMS_res).transpose()
    return est_im.reshape((sourceImg.shape[0], sourceImg.shape[1], 3));

def initialize(src):
    sourceImg = src / 255
    sourceRGB = np.rollaxis(sourceImg, -1)
    RGBarray = arrayRGB(sourceRGB)
    source_img = np.hstack((sourceRGB[0], sourceRGB[1], sourceRGB[2]))

    return sourceImg, sourceRGB, RGBarray,source_img

def staticalAlignment(p1,source):
    lab = np.dot(p1, source)
    mean = np.mean(lab, axis=1)
    std = np.std(lab, axis=1)
    return mean,std


def arrayRGB(sourceRGB):
    RGB_array = [[] for _ in range(3)]
    for i in  range(len(sourceRGB)):
        RGB_array[i] =  sourceRGB[i].reshape((sourceRGB[i].shape[0] * sourceRGB[i].shape[1], 1))

    return RGB_array

def zeros2one(array):
    zeros_indicies = np.where(array == 0)[0]
    array[zeros_indicies] = 1.0
    loggedSoruce = np.where(array > 0.0000000001, np.log10(array), -10)
    return array,loggedSoruce
