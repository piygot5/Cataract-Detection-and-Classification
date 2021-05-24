import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

url = 'http://192.168.0.103:5000'
#cap = cv2.VideoCapture(url)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(url)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
