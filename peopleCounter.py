import numpy as np
import cv2

cap = cv2.VideoCapture("peopleCounter.avi")

out = cv2.VideoWriter("test.avi", -1, 10.0, (640,480))


bgSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)


w = cap.get(3)
h = cap.get(4)

mx = int(w/2)
my = int(h/2)

count = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    bgMask = bgSub.apply(frame)
    try:
        #cv2.putText(frame, "count: {}".format(count), (mx,my), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1,1)
        cv2.imshow("Frame", bgMask)
        count += 1
    except Exception as e:
        print(e)
        print("EOF")
        break

    if(cv2.waitKey(30) & 0xFF == 27):
        break

cap.release()
cv2.destroyAllWindows()
