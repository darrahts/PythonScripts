import cv2
import numpy as np
import Person
import time

#   capture from file
cap = cv2.VideoCapture("peopleCounter.avi")

#   capture from stream
#cap = cv2.VideoCapture(0)


persons = []

cnt_up = 0
cnt_down = 0

max_p_age = 5
pid = 1
font = cv2.FONT_HERSHEY_SIMPLEX

subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=192, detectShadows=True)

kernel3 = np.ones((3,3),np.uint8)
kernel5 = np.ones((5,5),np.uint8)
kernel9 = np.ones((9,9),np.uint8)
kernel11 = np.ones((11,11),np.uint8)
kernel16 = np.ones((16,16), np.uint8)


while(cap.isOpened()):
    try:
        ret, frame = cap.read()
        mask = subtractor.apply(frame)
        ret, imgBin = cv2.threshold(mask,200,255,cv2.THRESH_BINARY)

        #imgMorphOpen = cv2.morphologyEx(imgBin, cv2.MORPH_OPEN, kernel3)
        imgMorphClose = cv2.morphologyEx(imgBin, cv2.MORPH_CLOSE, kernel5)
        
    except:
        print("EOF")
        break

    _, contours, hierarchy = cv2.findContours(imgMorphClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours = cv2.findContours(imgMorphClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    for cnt in contours:
        #contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if(area > 1600 and area < 5800):
            #print(area)
            moment = cv2.moments(cnt)
            try:
                center = (int(moment["m10"]/moment["m00"]), int(moment["m01"]/moment["m00"]))
                x,y,w,h = cv2.boundingRect(cnt)
                #cv2.drawContours(frame, cnt, -1, (0,255,0), 3, 8)
                cv2.circle(frame, (int(center[0]), int(center[1])), 3, (0,255,0), 3)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),1)

                new = True
                for p in persons:
                    if(abs(x-p.getX()) <= w and abs(y-p.getY()) <= h):
                        new = False
                        p.updateCoords(center[0], center[1])
                        break

                if(new == True):
                    p = Person.MyPerson(pid, center[0], center[1], max_p_age)
                    persons.append(p)
                    pid += 1

                for p in persons:
                    if(len(p.getTracks()) >= 2):
                        pts = np.array(p.getTracks(), np.int32)
                        pts = pts.reshape((-1,1,2))
                        frame = cv2.polylines(frame, [pts], False, p.getRGB())
                        #if(p.getId() == 9):
                        #    print(str(p.getX()), ",", str(p.getY()))
                        cv2.putText(frame, str(p.getId()), (p.getX()+10,p.getY()+10), font, 0.8, (0,255,255), 1, cv2.LINE_AA)


            except:
                print("pass")
                pass

        cv2.imshow("Frame", frame)
    
    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break
    
cap.release()
cv2.destroyAllWindows()
print()
print()
print("*******************************************")
print("number of persons: ", str(pid - 1))
print("*******************************************")
























