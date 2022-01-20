from sys import argv

from sys import *
import cv2
import pa1_2 as FS


def readGScale():
    return cv2.imread("dithering/" + argv[1], 0)

def quantizeImage(image, qVal):
    height, width = image.shape

    for i in range(height):
        for j in range(width):
            image[i][j] = int(qVal * (image[i][j] / 255)) * (255 / qVal)
    cv2.imwrite("dithering/" + "q= " + argv[4] + " - " + (argv[2]), image)
    return image


def showImage(image, text):
    cv2.imshow(text, image)

def main():
    #for first picture and quantized image
    q_value = int(argv[4])
    firstImage = readGScale()
    img = cv2.resize(firstImage, (320, 250))
    showImage(img, "Original Image")
    img = quantizeImage(img, q_value)
    showImage(img, "Quantized Image")

    # second part of code
    secondImage = readGScale()
    img2 = FS.FloydSteinberg(secondImage, q_value)
    img2 = cv2.resize(img2, (320, 250))
    showImage(img2, "FloydSteinberg Image")
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

