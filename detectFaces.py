import cv2
import matplotlib.pyplot as plt
import time
plt.ioff()

def makeRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

test1 = cv2.imread("dataset\\neutral\\S005_001_00000001.png")
if test1.any() == None:
    raise Exception("could not load image !")
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
cv2.imshow("Test Image", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()