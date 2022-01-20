import numpy as np
import cv2
import sys

def main(img,quantizedVal):
    FloydSteinberg(img,quantizedVal)

def FloydSteinberg(img, quantizedVal):
    width, height = img.shape

    for y in range(height - 1):
        for x in range(width - 1):
            old_pixel = img[x][y]
            new_pixel = find_quantized_value(old_pixel, quantizedVal)
            img[x][y] = new_pixel
            error = old_pixel - new_pixel
            img[x + 1][y] = (boundValue(img[x + 1][y], error, (7 / 16)))
            img[x - 1][y + 1] = (boundValue(img[x - 1][y + 1], error, (3 / 16)))
            img[x][y + 1] = (boundValue(img[x][y + 1], error, (5 / 16)))
            img[x + 1][y + 1] = (boundValue(img[x + 1][y + 1], error, (1 / 16)))


    #save images to file
    cv2.imwrite("ResultFolder/" + "q = " + sys.argv[4] + " - " + sys.argv[3], img)
    return img
def find_quantized_value(oPix, qVal):
    return int(qVal * (oPix / 255)) * (255 / qVal)


def boundValue(pixVal,err, frac):
    if pixVal + int(err * frac) < 0:
        return  0
    elif pixVal + int(err * frac) > 255:
        return 255
    else:
        return (err * frac)+pixVal
    return None
if __name__ == '__main__':
    main()
