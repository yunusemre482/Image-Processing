import numpy as np
from PIL import Image, ImageFilter
import os

def walkPictures():
    for  dirpath, dnames, fnames in os.walk("./Images"):
        for f in fnames:
            for var in [3,5,7,9]:
                for var in [9]:
                    # Example 1
                    print("---stage  start ---")
                    img = Image.open('Images/'+f)
                    result_mean = MeanFilter(img.copy(), var)
                    result_mean.save("./Results/image1_" + str(var) + ".jpg")

                    result_guas = GaussianFilter(img.copy(), var, 1)
                    result_guas.save("./Results/balloon_Gaussian_" + str(var) + ".jpg")

                    result_kuwahara = KuwaharaFilter('Images/'+f, var)
                    result_kuwahara.save("./Results/balloon_Kuwahara_" + str(var) + ".jpg")
                    print("---stage  end ---")


def MeanFilter(image, windowSize):
    width, height = image.size
    kVal = windowSize//2

    width ,height= width - kVal, height - kVal

    for i in range(kVal, width):
        for j in range(kVal, height):
            result_R, result_G, result_B = 0, 0, 0
            for k in range(-kVal, kVal + 1):
                for l in range(-kVal, kVal + 1):
                    result_R = result_R + image.getpixel((i + k, j + l))[0]
                    result_G = result_G + image.getpixel((i + k, j + l))[1]
                    result_B = result_B + image.getpixel((i + k, j + l))[2]
            result_R, result_G, result_B = int(result_R / (windowSize ** 2)), int(result_G / (windowSize ** 2)), int(
                result_B / (windowSize ** 2))
            image.putpixel((i, j), (result_R, result_G, result_B))
    return image


def GaussianFilter(image, windowSize, sigma):
    shapeCon = int((windowSize - 1) / 2)

    gaussian_mask = np.zeros((windowSize, windowSize))
    for i in range(-shapeCon, shapeCon + 1):
        for j in range(-shapeCon, shapeCon + 1):
            var_x = sigma ** 2 * (2 * np.pi)
            var_y = np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
            gaussian_mask[i + shapeCon, j + shapeCon] = (1 / var_x) * var_y

    width, height = image.size
    width = int(width - shapeCon)
    height = int(height - shapeCon)
    for i in range(shapeCon, width):
        for j in range(shapeCon, height):
            result_R, result_G, result_B = 0, 0, 0
            for k in range(-shapeCon, shapeCon + 1):
                for l in range(-shapeCon, shapeCon + 1):
                    result_R = (result_R + (image.getpixel((i + k, j + l))[0] * gaussian_mask[k + shapeCon][l + shapeCon]))
                    result_G = (result_G + (image.getpixel((i + k, j + l))[1] * gaussian_mask[k + shapeCon][l + shapeCon]))
                    result_B = (result_B + (image.getpixel((i + k, j + l))[2] * gaussian_mask[k + shapeCon][l + shapeCon]))
            image.putpixel((i, j), (int(result_R), int(result_G), int(result_B)))
    return image


def KuwaharaFilter(image_path, filter_dimension):
    RGB_image = np.array(Image.open(image_path), dtype=float)
    HSV_image = np.array(Image.open(image_path).convert("HSV"), dtype=float)
    image_width, image_height, image_channel = HSV_image.shape
    padding_value = filter_dimension // 2
    for row in range(padding_value, image_width - padding_value):
        for column in range(padding_value, image_height - padding_value):
            image_part = HSV_image[row - padding_value: row + padding_value + 1,
                         column - padding_value: column + padding_value + 1, 2]
            width, height = image_part.shape
            Q1 = image_part[0: height // 2 + 1, width // 2: width]
            Q2 = image_part[0: height // 2 + 1, 0: width // 2 + 1]
            Q3 = image_part[height // 2: height, 0: width // 2 + 1]
            Q4 = image_part[height // 2: height, width // 2: width]
            stds = np.array([np.std(Q1), np.std(Q2), np.std(Q3), np.std(Q4)])
            min_std = stds.argmin()

            if min_std == 0:
                RGB_image[row][column][0] = np.mean(
                    RGB_image[row - height // 2: row + 1, column: column + width // 2 + 1, 0])
                RGB_image[row][column][1] = np.mean(
                    RGB_image[row - height // 2: row + 1, column: column + width // 2 + 1, 1])
                RGB_image[row][column][2] = np.mean(
                    RGB_image[row - height // 2: row + 1, column: column + width // 2 + 1, 2])

            elif min_std == 1:
                RGB_image[row][column][0] = np.mean(
                    RGB_image[row - height // 2: row + 1, column - width // 2: column + 1, 0])
                RGB_image[row][column][1] = np.mean(
                    RGB_image[row - height // 2: row + 1, column - width // 2: column + 1, 1])
                RGB_image[row][column][2] = np.mean(
                    RGB_image[row - height // 2: row + 1, column - width // 2: column + 1, 2])

            elif min_std == 2:
                RGB_image[row][column][0] = np.mean(
                    RGB_image[row: row + height // 2 + 1, column - width // 2: column + 1, 0])
                RGB_image[row][column][1] = np.mean(
                    RGB_image[row: row + height // 2 + 1, column - width // 2: column + 1, 1])
                RGB_image[row][column][2] = np.mean(
                    RGB_image[row: row + height // 2 + 1, column - width // 2: column + 1, 2])

            elif min_std == 3:
                RGB_image[row][column][0] = np.mean(
                    RGB_image[row: row + height // 2 + 1, column: column + height // 2 + 1, 0])
                RGB_image[row][column][1] = np.mean(
                    RGB_image[row: row + height // 2 + 1, column: column + height // 2 + 1, 1])
                RGB_image[row][column][2] = np.mean(
                    RGB_image[row: row + height // 2 + 1, column: column + height // 2 + 1, 2])

    return Image.fromarray(RGB_image.astype(np.uint8))


if __name__ == '__main__':
    walkPictures()
