import cv2
import numpy as np
import glob
import random

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #initialize emotions

fisherFace = cv2.face.FisherFaceRecognizer_create() #initialize fisher face classifier

def getFiles(emotion): #get images files for the emotion, randomize it, split it 90:10
    files = glob.glob("dataset\\" + emotion + "\\*")
    random.shuffle(files)
    training = files[:-int(len(files) * 0.1)]
    prediction = files[-int(len(files) * 0.1):]
    return training, prediction

def organizeImages(): #read image from files, grayscale it, organize into data nad labels
    trainingData = []
    trainingLabels = []
    predictionData = []
    predictionLabels = []
    for emotion in emotions:
        trainingFiles, predictionFiles = getFiles(emotion)
        for file in trainingFiles:
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            trainingData.append(image)
            trainingLabels.append(emotions.index(emotion))
        for file in predictionFiles:
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            predictionData.append(image)
            predictionLabels.append(emotions.index(emotion))
    return trainingData, trainingLabels, predictionData, predictionLabels

def recognizeEmotion():
    trainingData, trainingLabels, predictionData, predictionLabels = organizeImages()
    print("Training Fisherface classifier")
    print("Size of training set: " + str(len(trainingData)) + " images")
    fisherFace.train(trainingData, np.asarray(trainingLabels))
    print("Predicting classification data")
    total = 0
    correct = 0
    incorrect = 0
    for image in predictionData:
        prediction, conf = fisherFace.predict(image)
        if prediction == predictionLabels[total]:
            correct += 1
            total += 1
        else:
            incorrect += 1
            total += 1
    percentCorrect = float((correct * 100) / total)
    return percentCorrect

subScore = []
for i in range(100):
    accuracy = recognizeEmotion()
    subScore.append(accuracy)
print("Accuracy is " + str(np.mean(subScore)) + "%")