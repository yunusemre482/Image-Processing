import cv2
import pa2_2

def main():
    sourceImage = cv2.imread("colortransfer/storm.jpg")
    targetImage = cv2.imread("colortransfer/ocean_sunset.jpg")

    result = pa2_2.colorTransfer(sourceImage, targetImage)
    cv2.imshow("source Image", sourceImage)
    cv2.imshow("target Image", targetImage)
    cv2.imshow("Result Image", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()

