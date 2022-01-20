import task as task
from PyQt5 import QtWidgets, uic, QtGui
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import main, time
from PyQt5.QtGui import QPainter
from PIL import ImageQt

class Ui(QtWidgets.QMainWindow):

    def task(self, task_fn):
        self.tasks[task_fn.__name__] = task_fn

    def __init__(self):
        super(Ui, self).__init__()

        uic.loadUi('frontend.ui', self)
        self.centralWidget=self.findChild(QtWidgets.QWidget,"centralwidget")
        self.grayscaled = False
        self.filepath = None
        self.detected = False
        self.saved = True
        self.image = None
        self.rubberband = None
        self.width, self.height = 0, 0
        self.cropCount = 1
        self.flag = False
        self.inputImage = None
        self.outputImage = None
        self.selectedCheckBox = None
        self.selectedValue = None
        self.rgbVal = None
        self.add = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        self.add.activated.connect(self.save)
        self.finish = QtWidgets.QAction("Quit", self)
        self.finish.triggered.connect(self.closeEvent)

        self.tasks = {"saveBtn": self.save, "loadBtn": self.load_image,
                      "deBlurBtn": self.deBlur, "grayScaleBtn": self.grayScaleImg, "flipCwBtn": self.flipCwBTn,
                      "flipCcwBtn": self.flipCCwBTn, "showBtn": self.show_image, "cropBtn": self.crop,
                      "mirrorBtn": self.mirror, "reverseBtn": self.reverse, "addNoiseBtn": self.addNoise,
                      "detectBtn": self.detect, "applyBtn": self.apply, "resetBtn": self.reset,
                      "brightness": self.changeBrightness, "saturation": self.changeSaturation,
                      "contrast": self.changeContrast,
                      "rotate": self.rotateImage, "blur": self.blurImage, "colorBalance": self.changeColorBalance
                      }
        self.boxes = {1: "brightnessValue", 2: "saturationValue", 3: "contrastValue", 4: "rotateValue", 5: "blurValue",
                      6: "colorBalanceValue"}

        self.groupBox = self.findChild(QtWidgets.QGroupBox, "groupBox")
        for checkBox in self.groupBox.children():
            checkBox.clicked.connect(self.selection(checkBox.objectName()))
        self.messageBox = self.findChild(QtWidgets.QLabel, "messageText")
        self.percentageBox = self.findChild(QtWidgets.QLabel, "percentageBox")

        for button in self.findChildren(QtWidgets.QPushButton):
            button.clicked.connect(self.process(button.objectName()))

            if button.objectName() == "loadBtn":
                pass
            else:
                button.setDisabled(True)

        self.progressBar = self.findChild(QtWidgets.QProgressBar, 'progressBar')  # Find the button
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)

        for valBox in self.findChildren(QtWidgets.QSpinBox):

            if valBox.objectName() == 'rotateValue':
                valBox.setRange(0, 360)
            elif valBox.objectName() == 'blurValue':
                valBox.setRange(0, 10)
            else:
                valBox.setRange(-255, 255)

        self.inputFrame = self.findChild(QtWidgets.QLabel, 'inputFrame')
        self.inputFrame.setScaledContents(True)
        self.inputFrame.setStyleSheet("QFrame {"
                                      "border-width: 1;"
                                      "border-radius: 3;"
                                      "border-style: solid;"
                                      "border-color: rgb(10, 10, 10)}"
                                      )

        self.outputFrame = self.findChild(QtWidgets.QLabel, 'outputFrame')
        self.outputFrame.setScaledContents(True)
        self.outputFrame.setStyleSheet("QFrame {"
                                       "border-width: 1;"
                                       "border-radius: 3;"
                                       "border-style: solid;"
                                       "border-color: rgb(10, 10, 10)}"
                                       )
        self.input = self.findChild(QtWidgets.QSpinBox, 'brightnessValue')
        self.setMouseTracking(False)
        self.input.valueChanged.connect(self.show_result)
        self.show()

    def selection(self, selectedName):
        def selectted():
            counter = 0
            index = None
            for checkBox in self.groupBox.children():
                if checkBox.objectName() != selectedName:
                    checkBox.setChecked(False)
                else:
                    index = counter
                counter += 1
            self.selectedValue = self.findChild(QtWidgets.QSpinBox, self.boxes[index + 1]).value()
            self.rgbVal = self.findChild(QtWidgets.QComboBox).currentText()
            self.selectedCheckBox = selectedName

        return selectted

    def process(self, btnName):
        def pressed():
            self.progressBar.setValue(0)
            self.percentageBox.setText("% 0")
            self.tasks[btnName]()
            for i in range(101):
                self.percentageBox.setText("% " + str(i))
                self.progressBar.setValue(i)

        return pressed


    def show_image(self):
        if self.outputImage is None:
            self.outputFrame.setPixmap(QtGui.QPixmap.fromImage(self.inputImage))
        else:
            self.outputFrame.setPixmap(QtGui.QPixmap.fromImage(self.outputImage))
        self.messageBox.setText("Message : The image has been shown the left side of frame succesfully")

        for self.button in self.findChildren(QtWidgets.QPushButton):
            if self.grayscaled:
                if self.button.objectName() == "deBlurBtn" or self.button.objectName() == "grayScaleBtn":
                    self.button.setDisabled(True)
                else:
                    self.button.setDisabled(False)
            else:
                self.button.setDisabled(False)
        for valBox in self.findChildren(QtWidgets.QSpinBox):
            if self.grayscaled:

                if valBox.objectName() in ["colorBalanceValue", "saturationValue", "contrastValue", "blurValue",
                                           "brightnessValue"]:
                    valBox.setDisabled(True)
            else:
                valBox.setDisabled(False)

        for checkBox in self.groupBox.children():
            if self.grayscaled:
                if checkBox.objectName() in ["colorBalance", "saturation", "contrast", "blur",
                                             "brightness"]:
                    checkBox.setDisabled(True)
            else:
                checkBox.setDisabled(False)

    def reset(self):
        qm = QtWidgets.QMessageBox
        ret = qm.question(self, 'Reset Changes', "Do you want to reset all changes?", qm.Yes | qm.No)
        if ret == qm.Yes:
            for checkBox in self.groupBox.children():
                checkBox.setChecked(False)
            for box in self.findChildren(QtWidgets.QSpinBox):
                box.setValue(0)
            self.grayscaled = False
            self.saved = False
            self.image = main.loadImage(self.filepath[0])
            self.width, self.height = self.image.shape[0], self.image.shape[1]
            self.image = main.resizeImage(self.image, 400, 400)
            self.outputImage = None
            self.inputImage = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0],
                                           QtGui.QImage.Format_RGB888).rgbSwapped()
            self.messageBox.setText("Message : The image has been uploaded succesfully")

            self.inputFrame.setPixmap(QtGui.QPixmap.fromImage(self.inputImage))
            self.rubberband = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self.inputFrame)
            self.setMouseTracking(True)
            self.show_image()
    def load_image(self):
        if not self.saved:
            qm = QtWidgets.QMessageBox
            ret = qm.question(self, 'Save', "Do you want to save the current Image?", qm.Yes | qm.No)
            if ret == qm.Yes:
                self.save()
        self.filepath = QtWidgets.QFileDialog.getOpenFileName(self, 'Hey! Select an Image',
                                                         filter="Image Files(*.jpeg;*.jpg;*.png)")
        if len(self.filepath[0])==0:
            self.messageBox.setText("Message : image can not uploaded")
            self.progressBar.setValue(100)


        if not len(self.filepath[0]) == 0:
            self.grayscaled = False
            self.saved = False
            self.image = main.loadImage(self.filepath[0])
            self.width, self.height = self.image.shape[0], self.image.shape[1]
            self.image = main.resizeImage(self.image, 400, 400)
            self.outputImage = None
            self.inputImage = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0],
                                           QtGui.QImage.Format_RGB888).rgbSwapped()
            self.messageBox.setText("Message : The image has been uploaded succesfully")

            self.inputFrame.setPixmap(QtGui.QPixmap.fromImage(self.inputImage))
            self.rubberband = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self.inputFrame)
            self.setMouseTracking(True)
            self.show_image()

    def changeColorBalance(self, values):
        if self.inputImage:
            if not self.grayscaled:
                self.outputImage = main.changeColorBalance(self.image, values[0], values[1])
                self.image = self.outputImage
                self.is_grayscaled()
                self.show_image()
                self.messageBox.setText("Message : The color balance changed succesfully")

    def changeContrast(self, values):
        if self.inputImage:
            if not self.grayscaled:
                self.outputImage = main.adjustContrastOfImage(self.image, values[0])
                self.image = self.outputImage
                self.is_grayscaled()
                self.show_image()
                self.messageBox.setText("Message : The contrast changed succesfully")

    def rotateImage(self, values):
        if self.inputImage:
            self.outputImage = main.rotateImage(self.image, values[0])
            self.image = self.outputImage
            self.is_grayscaled()
            self.show_image()
            self.messageBox.setText("Message : The image is rotated succesfully")

    def changeSaturation(self, values):
        if self.inputImage:
            if not self.grayscaled:
                self.outputImage = main.adjustSaturationOfImage(self.image, values[0])
                self.image = self.outputImage
                self.is_grayscaled()
                self.show_image()
                self.messageBox.setText("Message : The saturation changed succesfully")

    def changeBrightness(self, values):
        if self.inputImage:
            if not self.grayscaled:
                self.outputImage = main.adjustBrightnessOfImage(self.image, values[0])
                self.image = self.outputImage
                self.is_grayscaled()
                self.show_image()
                self.messageBox.setText("Message : The brightness changed succesfully")

    def save(self):
        try:
            filename = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image", filter="Image Files(*.jpg;*png;*.jpeg)")
            self.image = main.resizeImage(self.image, self.height, self.width)
            main.saveImage(self.image, filename[0])
            self.saved = True
            self.messageBox.setText("Message : The image saved")

        except:
            self.messageBox.setText("Message : The image can not saved succesfully!")

    def blurImage(self, values):
        if self.inputImage:
            if not self.grayscaled:
                self.outputImage = main.blurImage(self.image, values[0])
                self.image = self.outputImage
                self.is_grayscaled()
                self.show_image()

                self.findChild(QtWidgets.QPushButton, "grayScaleBtn").setDisabled(True)
                self.messageBox.setText("Message : The image has been blured succesfully")

    def deBlur(self):
        if not self.grayscaled:
            self.outputImage = main.deblurImage(self.image)
            self.image = self.outputImage
            self.outputImage = QtGui.QImage(self.outputImage.data, self.outputImage.shape[1], self.outputImage.shape[0],
                                            QtGui.QImage.Format_RGB888).rgbSwapped()

            self.show_image()
            self.findChild(QtWidgets.QPushButton, "grayScaleBtn").setDisabled(False)
            self.messageBox.setText("Message : The image has been deblured succesfully")

    def grayScaleImg(self):
        if not self.grayscaled:
            self.outputImage = main.grayScaleImage(self.image)

            self.image = self.outputImage
            self.outputImage = QtGui.QImage(self.outputImage.data, self.outputImage.shape[1], self.outputImage.shape[0],
                                            QtGui.QImage.Format_Grayscale8)
            self.grayscaled = True
            self.show_image()
            self.messageBox.setText("Message : The image has been gray scaled succesfully")
        else:
            self.messageBox.setText("Message : Already grayscaled")

    def mirror(self):
        self.outputImage = main.mirrorImage(self.image)
        self.image = self.outputImage
        self.is_grayscaled()
        self.show_image()
        self.messageBox.setText("Message : Mirror filter applied succesfully")

    def reverse(self):

        self.outputImage = main.reverseColor(self.image)
        self.image = self.outputImage
        self.is_grayscaled()
        self.show_image()
        self.messageBox.setText("Message : Reverse filter applied succesfully")

    def flipCwBTn(self):
        self.outputImage = main.flipImageClockWise(self.image)
        self.image = self.outputImage
        self.is_grayscaled()
        self.show_image()
        self.messageBox.setText("Message : The image has been flipped to clock wise direction succesfully")

    def flipCCwBTn(self):
        self.outputImage = main.flipImageCounterClockWise(self.image)
        self.image = self.outputImage
        self.is_grayscaled()
        self.show_image()
        self.messageBox.setText("Message : The image has been flipped to counter clock wise direction succesfully")

    def crop(self):
        self.cropCount+=1
        if(self.cropCount==2):
            self.rubberband.show()
            self.show_image()
            self.messageBox.setText("Message : The image crop succesfully")
        self.cropCount=1

    def mousePressEvent(self, event):
        if self.rubberband ==None:
            print("disabaled mouse event")
        elif (event.x() >= 50 and event.x() <= 450 and event.y() <= 701 and event.y() >= 300):
            self.origin = self.inputFrame.mapToGlobal(self.centralWidget.mapFromGlobal(event.pos()))
            self.rubberband.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
        else:
            pass

    def mouseMoveEvent(self, event):
        if self.rubberband ==None:
            print("disabaled mouse event")
        elif self.rubberband.isVisible():
            # Control the Rubber within the imageViewer!!!
            self.rubberband.setGeometry(QtCore.QRect(self.origin, event.pos()).normalized() & self.inputFrame.rect())
        else:
            pass


    def cropImage(self, rect):
        self.outputImage = self.inputImage.copy(rect)
        self.outputFrame.setPixmap(QtGui.QPixmap.fromImage(self.outputImage))
        self.update()

    def mouseReleaseEvent(self, event):
        if self.rubberband ==None:
            print("disabaled mouse event")
        elif self.rubberband.isVisible():
            self.rubberband.hide()
            selected = []
            rect = self.rubberband.geometry()

            self.cropImage(rect)
        else:
            pass
    def apply(self):

        counter = 0
        index = None
        try:
            for checkBox in self.groupBox.children():
                if checkBox.objectName() != self.selectedCheckBox:
                    print(end="")
                else:
                    index = counter
                counter += 1
            self.selectedValue = self.findChild(QtWidgets.QSpinBox, self.boxes[index + 1]).value()
            self.rgbVal = self.findChild(QtWidgets.QComboBox).currentText()
            self.tasks[str(self.selectedCheckBox)]([self.selectedValue, self.rgbVal])
            self.messageBox.setText("Message : The filter applied succesfully")
        except:
            print("Error")



    def addNoise(self):
        self.outputImage = main.addNoiseToImage(self.image, self.grayscaled)
        self.image = self.outputImage
        self.is_grayscaled()
        self.show_image()
        self.messageBox.setText("Message : Noise added succesfully")

    def detect(self):
        if not self.detected:
            self.outputImage = main.detectEdgesOfImage(self.image)
            self.image = self.outputImage
            self.grayscaled = True
            self.detected = True
            self.is_grayscaled()
            self.show_image()
            self.messageBox.setText("Message : All edges detected succesfully")
        else:
            self.messageBox.setText("Message : All edges already detected ")
    def show_result(self):
        pass

    def closeEvent(self, event):
        if not self.saved:
            qm = QtWidgets.QMessageBox
            ret = qm.question(self, 'Quit', "Do you want to save the current Image before Quit?", qm.Yes | qm.No)
            if ret == qm.Yes:
                self.save()
        event.accept()

    def is_grayscaled(self):
        if self.grayscaled:
            self.outputImage = QtGui.QImage(self.outputImage.data, self.outputImage.shape[1], self.outputImage.shape[0],
                                            QtGui.QImage.Format_Grayscale8)
        else:
            self.outputImage = QtGui.QImage(self.outputImage.data, self.outputImage.shape[1], self.outputImage.shape[0],
                                            QtGui.QImage.Format_RGB888).rgbSwapped()

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()
