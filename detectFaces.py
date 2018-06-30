import cv2

def makeRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

test1 = cv2.imread("dataset\\neutral\\S005_001_00000001.png")
if test1.any() == None:
    raise Exception("could not load image !")
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
haar_face_cascade = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_alt.xml")
haar_face_cascade_alt = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_alt_tree.xml")
haar_face_cascade_alt2 = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_alt2.xml")
haar_face_cascade_alt3 = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_default.xml")
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
if len(faces) < 1:
    faces = haar_face_cascade_alt.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    if len(faces) < 1:
        faces = haar_face_cascade_alt2.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
        if len(faces) < 1:
            faces = haar_face_cascade_alt3.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
            if len(faces) < 1:
                raise Exception("Could not find a face!")
for (x, y, w, h) in faces:
    cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Test Image', makeRGB(test1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()