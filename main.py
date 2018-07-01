import cv2

webcamCapture = cv2.VideoCapture(0)

haar_face_cascade = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_alt.xml") #initialize cascade classifier training
haar_face_cascade_alt = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_alt_tree.xml")
haar_face_cascade_alt2 = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_alt2.xml")
haar_face_cascade_alt3 = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_default.xml")

def extractFace(frame):
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

def detect():
    while True:
        captureSuccess, frame = webcamCapture.read() #Get frame from webcam; captureSuccess is true if success
        if captureSuccess == True:
            frameGS = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8)) #create CLAHE object
            improvedGSframe = clahe.apply(frameGS) #apply CLAHE to g/s frame from webcam, improve contrast
            faces = extractFace(improvedGSframe)
            if len(faces) < 1:
                pass
            else:
                for x, y, w, h in faces:
                    croppedFace = improvedGSframe[y:y+h, x:x+w]
                    if len(faces) == 1:
                        cv2.imshow("Detected face #1", croppedFace)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.imshow("Webacam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): #imshow expects a termination definition to work correctly, here it is bound to key 'q'
                break
        else:
            print("Couldn't grab frame from webcam!")

detect()