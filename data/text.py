from cgi import print_form
from multiprocessing.connection import Listener
from sre_constants import SUCCESS
from traceback import print_tb
from matplotlib.pyplot import box
import cv2
import numpy as np
import listener as listener
import speech_recognition as sr
import pyttsx3
import AI_Properties as AP
from FindObjectClass import FindObjectClass
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

#Function Command KeyWords
def run_identifeye():
    command = waitCommand()
    print(command)
    if AIN._CandyName in command:
        talk('okay, what do you want to look for?')
    else:
        talk('I did not get that, please say again.')
        run_identifeye()

#Open Application Greet
def OpenApp_Greet(): 
    talk('Yes???')
    run_identifeye()
#Execute OpenApp_Greet First
OpenApp_Greet()

#Function command following response
def ffResponse():
    talk('okay, what do you want to look for?')
    ff_command()
#Function Incorrect input command
def ffIncorrect():
    print()
#Function following command
def ff_command(item): 
    com = waitCommand()
    print(com)
    if com == classNames:
        print(f"Found{com}")
    else:
        print(f"Cannot found{com}")
        ffQuestion()
    return item+com
    

#Question for Object
def ffQuestion(): 
    cv2.destroyAllWindows()
    talk('is there anything else you are looking for?')
    ff = waitCommand()
    if 'yes' in ff:
        ffResponse()
    elif 'no' in ff:
        talk('okay, thank you. goodbye')
        exit()
    else:
        talk('sorry, I did not get that.')
        ffQuestion()

run_identifeye()

#Match Specific ID object command
checkpoint = ff_command(item='')
print(checkpoint)
objn1 = FindObjectClass(checkpoint)
print(objn1.name)

def idef():
    return objn1.name

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
        #if idef() == className[classId - 1]:
        #        talk('I found a' + idef())
        #        ffQuestion()
        
        
       
        
    
    
    cv2.imshow("Output Test",img)
    cv2.waitKey(1)

