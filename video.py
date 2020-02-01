import cv2
import numpy as np
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

face_classifier = cv2.CascadeClassifier("Haar_cascade/haarcascade_frontalface_default.xml")

def face_detector(img):
    # Convert image to grayscale
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=img
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0,0,0,0), np.zeros((128,128,3), np.uint8), img

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]

    try:
        roi_gray = cv2.resize(roi_gray, (128, 128), interpolation = cv2.INTER_AREA)
    except:
        return (x,w,y,h), np.zeros((128,128), np.uint8), img
    return (x,w,y,h), roi_gray, img

cap = cv2.VideoCapture(0)
class_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
classifier = load_model("models/Kaggle_model.h5")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter("Videos/face.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
while True:

    ret, frame = cap.read()
    rect, face, image = face_detector(frame)
    if np.sum([face]) != 0.0:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # make a prediction on the ROI, then lookup the class
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]
        label_position = (rect[0] + int((rect[1]/2)), rect[2] + 25)
        cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
    else:
        cv2.putText(image, "No Face Found", (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)

    cv2.imshow('All', image)
    #out.write(image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
