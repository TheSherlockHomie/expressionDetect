import cv2
import glob

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #define emotions

haar_face_cascade = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_alt.xml") #initialize cascade classifier training
haar_face_cascade_alt = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_alt_tree.xml")
haar_face_cascade_alt2 = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_alt2.xml")
haar_face_cascade_alt3 = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_default.xml")

def extractFace(emotion):
    files = glob.glob("dataset\\" + emotion + "\\*")
    fileNumber = 0
    for x in files:
        image = cv2.imread(x) #read an image
        if image.any() == None:
            raise Exception("Could not read Image " + x)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #make it grayscale, performance reasons
        face = haar_face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5) #get face
        if len(face) < 1:
            face = haar_face_cascade_alt.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
            if len(face) < 1:
                face = haar_face_cascade_alt2.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
                if len(face) < 1:
                    face = haar_face_cascade_alt3.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
                    if len(face) < 1:
                        raise Exception("Could not find a face!") #find face at any cost
        for x, y, w, h in face: #get face coordinates
            image = image[y:y+h, x:x+w] #crop face
            image = cv2.resize(image, (350, 350)) #Equalize face dimensions
            cv2.imwrite("dataset\\" + emotion + "\\" + str(fileNumber) + ".jpg", image) #save extracted face
            fileNumber += 1

for emotion in emotions:
    extractFace(emotion) #actually run code