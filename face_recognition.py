import cv2
import numpy as np
import os
from datetime import datetime
import gspread
import csv
from oauth2client.service_account import ServiceAccountCredentials
import sys
import RPi.GPIO as GPIO
from time import sleep
import urllib3
import board
import busio as io
import adafruit_mlx90614

#code for google sheets and thingspeak
baseURL='http://api.thingspeak.com/update?api_key=SJXRR64R4EUWJ44X&field1='
http=urllib3.PoolManager()
scope=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds=ServiceAccountCredentials.from_json_keyfile_name('mydata-cddcc1b90525.json',scope)
client=gspread.authorize(creds)
sheet=client.open("Attendance Final").worksheets()

#code for initializing temperature sensor
i2c = io.I2C(board.SCL, board.SDA, frequency=100000)
mlx = adafruit_mlx90614.MLX90614(i2c)

#code for initializing face recognition system
recognizer= cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/pi/trainer/trainer.yml')
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml');
font=cv2.FONT_HERSHEY_SIMPLEX

#function to upload values
def upload(id, id1,c,targetTemp):
    for ii in sheet:
        ii.append_row([str(c),datetime.today().strftime('%Y-%m-%d'),datetime.today().strftime(' %H:%M:%S'),str(id),str(id1), str(targetTemp)+" °C"])
        print("Updated attendance for ", id1 )
    flag[id]=False
    f=http.request('GET', baseURL+str(c))
    f.read()
    f.close()
    sleep(5)
    print(str(c)+ " people in room currently")
    
    
#initializing values        
id=0
c=1

names=['None', 'Namitha', 'Cal', 'Tae']
flag=[True,True,True,True]
cam=cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

minW=0.1*cam.get(3)
minH=0.1*cam.get(4)

while True:
    ret,img=cam.read()
    img=cv2.flip(img,-1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5, minSize=(int(minW),int(minH)),)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
        id,confidence=recognizer.predict(gray[y:y+h,x:x+w])
        if(confidence<100):
            id1=names[id]
            if(flag[id]):
                confidence=" {0}%".format(round(100-confidence))
                print ("Hold out hand to record temperature")
                sleep(5)
                targetTemp = "{:.2f}".format(mlx.object_temperature)
                upload(id,id1,c,targetTemp)
                c+=1
                #file=open("attendance.csv","a")
                #flag[id]=False
                #file.write(str(c)+ "," + datetime.today().strftime('%Y-%m-%d')+","+datetime.today().strftime(' %H:%M:%S') + "," +str(id) + "," + str(id1) +"\n" )
                #print("\n Attendance marked for " + id1 + "\n" + str(c) +" people in room currently")
                #c+=1
                #file.close()
        else:
            id1="unknown"
            confidence=" {0}%".format(round(100-confidence))
        cv2.putText(img, str(id1), (x+5,y-5), font,1,(255,255,0),1)
    cv2.imshow('camera',img)
    k=cv2.waitKey(10) & 0xff
    if k==27:
        break

print("\n [INFO Exiting Program and cleanup stuff")
cam.release()

cv2.destroyAllWindows()

                                   