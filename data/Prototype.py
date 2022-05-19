from multiprocessing.connection import Listener
from sre_constants import SUCCESS
from threading import Thread
from traceback import print_tb
from matplotlib.pyplot import box
import cv2
import numpy as np
import listener as listener
import speech_recognition as sr
import pyttsx3
import AI_Properties as AP
from FindObjectClass import FindObjectClass
import time as clock
#Open CV properties
thres = 0.5

nsm_threshold = 0.2

cap = cv2.VideoCapture(0)
cap.set(3, 648)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -2.0)
cap.set(cv2.CAP_PROP_EXPOSURE, -2.0)

classNames= []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

print("Excuting please wait..")

#Speech Recognition
listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def talk(text):
    engine.say(text)
    engine.runAndWait()

#Encapsulation Properties PATH AI_Properties.py

AIN = AP.Properties()
print(AIN._CandyName)

#Take Command
def waitCommand():
    try:
        with sr.Microphone() as source:
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            if AIN._CandyName in command:
                command = command.replace(AIN._CandyName)
                print(command)
    except:
        pass
    return command



#Function command following response
def ffResponse():
    talk('okay, what do you want to look for?')
    print("okay, what do you want to look for?")
    ff_command()

#Question for Object
def ffQuestion(): 
    cv2.destroyAllWindows()
    talk('is there anything else you are looking for?')
    ff = waitCommand()
    if 'yes' in ff:
        ffResponse()
    elif 'no' in ff:
        talk('okay, thank you. goodbye')
        print('okay, thank you. goodbye')
        exit()
    else:
        talk('sorry, I did not get that.')
        print('sorry, I did not get that.')
        ffQuestion()


#Function Incorrect input command
def ffIncorrect():
    print()
    
#Open Application Greet
def OpenApp_Greet(): 
    talk('Yes???')
    run_identifeye()


#Function following command
#Match Specific ID object command
def ff_command():
    print("Listening...")
    com = waitCommand()
    print(com)
    objn1 = FindObjectClass(com)
    
    def idef():
        return objn1.name
    def time():
        clock.sleep(0.5)

            #OpenCV
    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1,-1).tolist()[0])
        confs = list(map(float, confs))

        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nsm_threshold)
        for i in indices:
             
            box = bbox[i]
            x,y,w,h = box[0],box[1],box[2],box[3]

            cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 0, 255), thickness=1)
            cv2.putText(img, classNames[classIds[i]-1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
            #talk(classNames[classIds[i]-1])
            talk(classNames[classIds[i]-1])
            clock.sleep(0.5)
                #if idef() == className[classId - 1]:
                #        talk('I found a' + idef())
                #        ffQuestion()
        
        
        cv2.imshow("Output Test",img)
        cv2.waitKey(1)



#Function Command KeyWords
def run_identifeye():
    command = waitCommand()
    print(command)
    if AIN._CandyName in command: #KeyWord
        talk('okay, what do you want to look for?')
        print('okay, what do you want to look for?')
        #OpenCV
        ff_command()
    else:
        talk('I did not get that, please say again.')
        print("I did not get that, please say again.")
        run_identifeye()

#Execute OpenApp_Greet First
OpenApp_Greet()