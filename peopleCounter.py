import numpy as np
import cv2

cap = cv2.VideoCapture(0)


fourcc = cv2.VideoWriter_fourcc(*'MJPG')

w = int(cap.get(3))
h = int(cap.get(4))
print(w,h)

out = cv2.VideoWriter("test2.avi", fourcc, 10.0, (w,h))


bgSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)


mx = int(w/2)
my = int(h/2)

count = 0

try:
    while(cap.isOpened()):
        ret, frame = cap.read()
        try:
##        if(cv2.waitKey(1) & 0xFF == ord('q')):
##            break
        #cv2.putText(frame, "count: {}".format(count), (mx,my), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1,1)
            if (ret == True):
                #print("true")
                frame = cv2.flip(frame,-1)
                bgMask = bgSub.apply(frame)
                gray = cv2.cvtColor(bgMask, cv2.COLOR_GRAY2RGB)
            #cv2.imshow("Frame", bgMask)
                out.write(gray)
        #count += 1
        except Exception as e:
            print(e)
            print("EOF")
            break
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("done")

