import cv2

webcamCapture = cv2.VideoCapture(0)

haar_face_cascade = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_alt.xml") #initialize cascade classifier training
haar_face_cascade_alt = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_alt_tree.xml")
haar_face_cascade_alt2 = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_alt2.xml")
haar_face_cascade_alt3 = cv2.CascadeClassifier("cascade-classifiers\haarcascade_frontalface_default.xml")

while True:
    captureYes, frame = webcamCapture.read() #Get frame from webcam; captureYes is true if success
    frameGS = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8)) #create CLAHE object
    improvedGSframe = clahe.apply(frameGS) #apply CLAHE to g/s frame from webcam, improve contrast
    faces = haar_face_cascade.detectMultiScale(improvedGSframe, scaleFactor=1.1, minNeighbors=5) #get faces
    if len(faces) < 1:
        faces = haar_face_cascade_alt.detectMultiScale(improvedGSframe, scaleFactor=1.1, minNeighbors=5)
        if len(faces) < 1:
            faces = haar_face_cascade_alt2.detectMultiScale(improvedGSframe, scaleFactor=1.1, minNeighbors=5)
            if len(faces) < 1:
                faces = haar_face_cascade_alt3.detectMultiScale(improvedGSframe, scaleFactor=1.1, minNeighbors=5)
                if len(faces) < 1:
                    pass
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow("Webcam Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #imshow expects a termination definition in order to work correctly, here it is bound to key 'q'
        break
