import imutils
import cv2
import math
from PIL import Image, ImageEnhance
import numpy as np
import random
from skimage.util import random_noise


def loadImage(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return image


def resizeImage(image, x, y):
    return cv2.resize(image, (x, y))


def saveImage(image, path):
    cv2.imwrite(path, image)
    success, buf = cv2.imencode(".jpg", cv2.resize(image, (image.shape[1], image.shape[0])))
    buf.tofile(path)


def blurImage(image, val):
    image = cv2.blur(image, (val, val))
    return image


def deblurImage(image):
    smoothed = cv2.GaussianBlur(image, (9, 9), 10)
    unsharped = cv2.addWeighted(image, 1.5, smoothed, -0.5, 0)
    return unsharped


def grayScaleImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def cropSelectedPartOfImage(image, startWidth, stopWidth, startHeight, stopHeight):
    image = image[startHeight:stopHeight, startWidth:stopWidth]
    return image


def flipImageClockWise(image):
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def flipImageCounterClockWise(image):
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def mirrorImage(image):
    image = cv2.flip(image, 1)
    return image


def rotateImage(image, degree):
    image = imutils.rotate(image, degree)
    return image


def reverseColor(image):
    image = (255 - image)
    return image


def changeColorBalance(image, balance, r_g_b):
    color = np.zeros_like(image)
    if balance > 255:
        balance = 255
    elif balance < 0:
        balance = 0
    if r_g_b == "Red":
        color[:, :] = [0, 0, balance]
    elif r_g_b == "Green":
        color[:, :] = [0, balance, 0]
    elif r_g_b == "Blue":
        color[:, :] = [balance, 0, 0]
    image = cv2.add(image, color)
    return image


def adjustBrightnessOfImage(image, brightness):
    brightness = mapValue(brightness, -255, 255, -255, 255)
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    else:
        buf = image.copy()

    return buf


def adjustSaturationOfImage(image, saturation):
    image = image.astype(np.float32) / 255.0
    hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hlsImg[:, :, 2] = (1.0 + saturation / float(100)) * hlsImg[:, :, 2]
    hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
    lsImg = lsImg.astype(np.uint8)
    return lsImg


def adjustContrastOfImage(image, contrast):
    contrast = mapValue(contrast, -100, 100, -80, 80)
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    return image


def addNoiseToImage(image, isGray):
    if not isGray:
        gauss = np.random.normal(0, 1, image.size)
        gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
        image = image + image * gauss
        return image
    else:
        image = random_noise(image, 's&p', amount=0.02)
        noise_img = np.array(255 * image, dtype='uint8')
        return noise_img


def detectEdgesOfImage(image):
    img_blur = cv2.GaussianBlur(image, (5, 5), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=img_blur, threshold1=0, threshold2=300)
    return edges


def mapValue(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
