import cv2
import numpy as np
from PIL import Image
import os

path='/home/pi/FacialRecognitionProject/dataset'
recognizer=cv2.face.LBPHFaceRecognizer_create()
detector=cv2.CascadeClassifier('/home/pi/Documents/lbpcascade_frontalface_improved.xml');
def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids=[]
    for imagePath in imagePaths:
        PIL_img=Image.open(imagePath).convert('L')
        img_numpy=np.array(PIL_img, 'uint8')
        id= int(os.path.split(imagePath)[-1].split(".")[1])
        faces=detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+h])
            ids.append(id)
    return faceSamples,ids
    
print ("\n [INFO] Training Faces. It will take a few seconds. Wait...")

faces,ids=getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer.write('/home/pi/trainer/trainer.yml')
print("\n [INFO] {0} faces trained. Exiting program".format(len(np.unique(ids))))