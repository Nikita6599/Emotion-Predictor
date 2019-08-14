from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing import image
import pandas as pd
import cv2
import numpy as np
import os
import time

labels = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}

dataframe=pd.read_csv('emotions_df.csv')


from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #load weights



img1=image.load_img('image4.jpg',target_size=(48,48),grayscale=True)
img1=image.img_to_array(img1)
img1=np.expand_dims(img1,axis=0)

answer=model.predict(img1)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
i=0
while True:
    time.sleep(0.35)
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray,1.5,5)
    
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=roi_gray.reshape(1,48,48,1)
        s = model.predict(roi_gray)
       
        ans=s.argmax()
        
        if ans==0:
            cv2.putText(frame,labels[0],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,cv2.LINE_AA)
        elif ans==1:
            cv2.putText(frame,labels[1],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,cv2.LINE_AA)
        elif ans==2:
            cv2.putText(frame,labels[2],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,cv2.LINE_AA)
        elif ans==3:
            cv2.putText(frame,labels[3],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,cv2.LINE_AA)
        elif ans==4:
            cv2.putText(frame,labels[4],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,cv2.LINE_AA)
        elif ans==5:
            cv2.putText(frame,labels[5],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,cv2.LINE_AA)
        elif ans==6:
            cv2.putText(frame,labels[6],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1,cv2.LINE_AA)
        
    cv2.imshow('Video',frame)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
