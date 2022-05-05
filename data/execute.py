import sys
import time
import cv2 as cv2
import listener as listener
import pyttsx3
import speech_recognition as sr
from FindObjectClass import FindObjectClass


thres = 0.5
cap = cv2.VideoCapture(0)
cap.set(3, 648)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -2.0)
cap.set(cv2.CAP_PROP_EXPOSURE, -2.0)

className = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    className = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

pTime = 0

listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


def talk(text):
    engine.say(text)
    engine.runAndWait()


def takeCommand():
    try:
        with sr.Microphone() as source:
            print('Listening...')
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            if 'identify' in command:
                command = command.replace('identify')
                print(command)
    except:
        pass
    return command



def run_identifeye():
        talk('Im Listening...')
        command = takeCommand()
        print(command)
        if 'identify' in command:
            talk('okay, what do you want to look for?')
        else:
            talk('I did not get that, please say again.')
            run_identifeye()


run_identifeye()


def ffResponse():
    talk('okay, what do you want to look for?')


def ffQuestion():
    cv2.destroyAllWindows()
    talk('is there anything else you are looking for?')
    ff = takeCommand()
    if 'yes' in ff:
        ffResponse()
    elif 'no' in ff:
        talk('okay, thank you. goodbye')
        exit()
    else:
        talk('sorry, I did not get that.')
        ffQuestion()


com = takeCommand()
objn1 = FindObjectClass(com)
print(objn1.name)


def idef():
    return objn1.name


# def detectObject():
#     detected, img = cap.read()
#     classIds, confs, bbox = net.detect(img, confThreshold=thres)
#
#     if len(classIds) != 0:
#         for classId, (confidence), box in zip(classIds.flatten(), confs.flatten(), bbox):
#             if idef() == className[classId - 1]:
#                 talk('I found a' + idef())
#                 break
#                 ffQuestion()


while True:
    detected, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    #print(className, bbox)

    if len(classIds) != 0:
        for classId, (confidence), box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 0, 255), thickness=1)
            cv2.putText(img, className[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
            cv2.putText(img, str(round(confidence * 100)) + "%", (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            if idef() == className[classId - 1]:
                talk('I found a' + idef())
                ffQuestion()

    # detectObject()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (1, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), int(1.5))

    cv2.imshow('Test', img)
    cv2.waitKey(1)


