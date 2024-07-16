import numpy as np

from sort import  *
from ultralytics import YOLO
import cvzone
import cv2

# cap = cv2.VideoCapture(0) # for webcam
# cap.set(3 ,480)
# cap.set(4 ,760)
cap = cv2.VideoCapture('../videos/cars.mp4')

signal_lengths = [20,30,40,50]
signal_time = signal_lengths[1]

tracker = Sort(max_age=20,min_hits=3,iou_threshold=.3)
model = YOLO('../Yolo-Weights/yolov8n.pt')


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread('Untitled design.png')

limits = [400,297,673,297]
totalCount = []


while True:
    success, img = cap.read()
    imregion = cv2.bitwise_and(img, mask)
    dets = np.empty((0, 5))
    results = model(imregion,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w,h = x2-x1, y2-y1


            conf = int(box.conf[0]*100)/100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == 'car' or currentClass == 'truck' or currentClass == 'bus' or currentClass == 'motorbike' and conf > 2:
                # cvzone.putTextRect(img,f"{currentClass}: {conf}", (x1,y1),thickness=0, scale=1)
                currntArray = np.array([x1, y1, x2, y2, conf])
                # cvzone.cornerRect(img, (x1, y1, w, h), l=7, rt=5)
                dets = np.vstack((dets,currntArray))

    # cv2.imshow('Image', imregion)
    resultsTracker = tracker.update(dets=dets)
    cv2.line(img,(limits[0], limits[1]), (limits[2], limits[3]),color=(0,0,255),thickness=3)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        # print(result)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img,(x1,y1,w,h), l=7,rt=2,colorR=(255,0,255))
        cvzone.putTextRect(img,f'{int(id)}', (x1,y1),thickness=3, scale=2)
        #calculcating center to see if it touches the line
        cx,cy = x1+w//2,y1+h//2
        # cv2.circle(img,(cx,cy), radius=5, color=(255,0,0))

        if limits[0]<cx<limits[2] and limits[1]-15<cy<limits[1]+15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 255, 0), thickness=3)
    cvzone.putTextRect(img, f"Count:{len(totalCount)}", (50, 50), thickness=0, scale=1)

    if len(totalCount) <20:
        signal_time = signal_lengths[0]
    elif len(totalCount) >= 20 and len(totalCount) <30:
        signal_time  = signal_lengths[1]
    elif len(totalCount) >= 30 and len(totalCount) < 40:
        signal_time  = signal_lengths[2]
    else:
        signal_time = signal_lengths[3]



    cv2.imshow('Image', img)
    print(signal_time)
    cv2.waitKey(0)