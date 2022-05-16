from cgi import print_form
from pydoc import classname
from sre_constants import SUCCESS
import cv2
from matplotlib.pyplot import box
import numpy as np
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
        
        
       
        
    
    
    cv2.imshow("Output Test",img)
    cv2.waitKey(1)

