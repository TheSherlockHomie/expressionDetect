import cv2
import numpy as np
import argparse
import time
import glob
import os

webcamCapture = cv2.VideoCapture(0)

haar_face_cascade = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_alt.xml") #initialize cascade classifier training
haar_face_cascade_alt = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_alt_tree.xml")
haar_face_cascade_alt2 = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_alt2.xml")
haar_face_cascade_alt3 = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_default.xml")

emotions = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"] #initialize emotions

fisherFace = cv2.face.FisherFaceRecognizer_create() #initialize fisher face classifier

# try:
#     fisherFace.load("trainedClassifier.xml")
# except:
#     print("Pre-trained classifier XML not found. Using --update will create one.")

parser = argparse.ArgumentParser(description="Options for Expression Detector") #Create parser object
parser.add_argument("--update", help="Call to grab new images and update the model accordingly", action="store_true") # Add --update argument
args = parser.parse_args() #Store any given arguments in an object

faceDict = {}

def getFiles(emotion): #get images files for the emotions
    files = glob.glob("dataset\\" + emotion + "\\*")
    training = files[:]
    return training

def organizeImages(): #read image from files, grayscale it, organize into data and labels
    trainingData = []
    trainingLabels = []
    for emotion in emotions:
        trainingFiles = getFiles(emotion)
        for file in trainingFiles:
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            trainingData.append(image)
            trainingLabels.append(emotions.index(emotion))
    return trainingData, trainingLabels

def trainClassifier():
    trainingData, trainingLabels = organizeImages()
    print("Training Fisherface classifier")
    print("Size of training set: " + str(len(trainingData)) + " images")
    fisherFace.train(trainingData, np.asarray(trainingLabels))

def getWebcamFrame():
    captureSuccess, frame = webcamCapture.read() #Get frame from webcam; captureSuccess is true if success
    if captureSuccess == True:
        frameGS = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8)) #create CLAHE object
        improvedGSframe = clahe.apply(frameGS) #apply CLAHE to g/s frame from webcam, improve contrast
        return improvedGSframe, frame
    else:
        raise Exception("Couldn't grab frame from webcam!")
        return ""

def detectFace(frame):
    faces = haar_face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5) #get faces
    if len(faces) < 1:
        faces = haar_face_cascade_alt.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
        if len(faces) < 1:
            faces = haar_face_cascade_alt2.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
            if len(faces) < 1:
                faces = haar_face_cascade_alt3.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
                if len(faces) < 1:
                    pass
    return faces

def predictEmotion(faces, frame):
    predictions = []
    for (x, y, w, h) in faces:
        currentFace = frame[y:y+h, x:x+w]
        currentFace = cv2.resize(currentFace, (350, 350))
        prediction, confidence = fisherFace.predict(currentFace)
        predictions.append(prediction)
    return predictions

def updateModel():
    print("Activated: Model update mode")
    checkDirStructure()
    for i in range(0, len(emotions)):
        saveFace(emotions[i])
    print("Collected images. Updating model")
    update()
    print("Done")

def update():
    trainClassifier()
    print("Saving model")
    try:
        fisherFace.write("trainedClassifier.xml")
    except:
        raise Exception("Error saving model")
    print("Model saved")

def checkDirStructure(): #check if folder infrastructure is there, create if absent
    for x in emotions:
        if os.path.exists("dataset\\" + x):
            pass
        else:
            os.makedirs("dataset\\" + x)

def saveFace(emotion):
    print("Pose with the emotion: " + emotion + " after the timer and freeze")
    for i in range(0,5): #Timer
        print(5-i)
        time.sleep(1.5)
    while len(faceDict.keys()) < 16: #Grab 15 images for each emotion
        frame, feed = getWebcamFrame()
        faces = detectFace(frame)
        if len(faces) == 1:
            for (x, y, w, h) in faces:
                croppedFace = frame[y:y+h, x:x+w]
                croppedFace = cv2.resize(croppedFace, (350, 350))
                faceDict["face" + str(len(faceDict)+1)] = croppedFace
        else:
            print("no/multiple faces detected, passing over frame")
    for x in faceDict.keys(): #save contents of dictionary to files
        try:
            cv2.imwrite("dataset\\" + emotion + "\\" + str(len(glob.glob("dataset\\" + emotion + "\\*")) + 1) + ".jpg", faceDict[x])
        except:
            raise Exception("Couldn't write file to disk")
    faceDict.clear() #clear dictionary so that the next emotion can be stored

def overlayEmoji(feed, faces, predictions):
    fNumber = 0
    for (x, y, w, h) in faces:
        emotion = emotions[predictions[fNumber]]
        emojiToDraw = cv2.imread("graphics\\" + emotion + ".png")
        feed = drawEmoji(feed, emojiToDraw, (x, y, w, h))
        fNumber += 1
    return feed

def drawEmoji(feed, emojiToDraw, coordinates):
    x, y, w, h = coordinates
    emojiToDraw = cv2.resize(emojiToDraw, (h, w))
    feed[y:y + h, x:x + w ] = emojiToDraw
    return feed

while True:
    if args.update: #If update flag is present, call update function
        updateModel()
        break
    trainClassifier()
    frame, feed = getWebcamFrame()
    faces = detectFace(frame)
    predictions = predictEmotion(faces, frame)
    feed = overlayEmoji(feed, faces, predictions)
    cv2.imshow("Expression Detection", feed) #Display frame
    if cv2.waitKey(1) & 0xFF == ord('q'): #imshow expects a termination definition to work correctly, here it is bound to key 'q'
        break