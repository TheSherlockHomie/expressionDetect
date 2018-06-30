import cv2
import matplotlib.pyplot as plt
import time
plt.ion()

def makeRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

test1 = cv2.imread("dataset\\neutral\\S005_001_00000001.png")
if test1.any() == None:
    raise Exception("could not load image !")
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
haar_face_cascade = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_alt.xml")
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
print(faces)
for (x, y, w, h) in faces:
    cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Test Image', makeRGB(test1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()