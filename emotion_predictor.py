import pandas as pd
import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing import image

labels = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}

df=pd.read_csv('C:/Users/Nikita/Desktop/Emoji/dataset/fer2013/fer2013.csv')

df.info()

m = [i for i in range(35000)]

dataframe = pd.DataFrame(index=m)

for i in range(35000):
    l =[]
    l = df['pixels'][i].split(' ')
    for _ in range(len(l)):
        dataframe.loc[i,'pixel_'+str(_)] = int(l[_])
        #print(dataframe['pixel_'+str(_)])
    print(i)
    
dataframe.head()

for i in range(35000):
    dataframe.loc[i,'emotions']= df.loc[i,'emotion']

dataframe.to_csv('emotions_df.csv')

x_train = dataframe.iloc[:,:-1].values
x_train = x_train.reshape(35000,48,48,1)
y_train = dataframe.iloc[:,-1].values
y_train = to_categorical(y_train)

model = Sequential() 
 
#1st convolution layer
model.add(Convolution2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
 
#2nd convolution layer
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
 
#3rd convolution layer
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
 
model.add(Flatten())
 
#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
 
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=30,batch_size=500,validation_split=0.2)

img1=image.load_img('image1.jpg',target_size=(48,48),grayscale=True)
img1=image.img_to_array(img1)
img1=np.expand_dims(img1,axis=0)

answer=model.predict(img1)


model.save('emotion_trainer.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
i=0
while True:
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
            cv2.putText(frame,labels[0],(x,y),cv2.FONT_HERSHEY_SIMPLEX,4,(0,255,0),1,cv2.LINE_AA)
        elif ans==1:
            cv2.putText(frame,labels[1],(x,y),cv2.FONT_HERSHEY_SIMPLEX,4,(0,255,0),1,cv2.LINE_AA)
        elif ans==2:
            cv2.putText(frame,labels[2],(x,y),cv2.FONT_HERSHEY_SIMPLEX,4,(0,255,0),1,cv2.LINE_AA)
        elif ans==3:
            cv2.putText(frame,labels[3],(x,y),cv2.FONT_HERSHEY_SIMPLEX,4,(0,255,0),1,cv2.LINE_AA)
        elif ans==4:
            cv2.putText(frame,labels[4],(x,y),cv2.FONT_HERSHEY_SIMPLEX,4,(0,255,0),1,cv2.LINE_AA)
        elif ans==5:
            cv2.putText(frame,labels[5],(x,y),cv2.FONT_HERSHEY_SIMPLEX,4,(0,255,0),1,cv2.LINE_AA)
        elif ans==6:
            cv2.putText(frame,labels[6],(x,y),cv2.FONT_HERSHEY_SIMPLEX,4,(0,255,0),1,cv2.LINE_AA)
        
    cv2.imshow('Video',frame)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
