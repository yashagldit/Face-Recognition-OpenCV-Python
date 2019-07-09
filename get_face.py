import cv2
import os

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
if not os.path.exists("img"):
    os.makedirs("img")
cap = cv2.VideoCapture(0)
x1 = 0
fold = input("Enter name")
pt = "img/" + fold
if not os.path.exists(pt):
    os.makedirs(pt)
while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y + h, x:x + h]
        roi_color = frame[y:y + h, x:x + h]
        img_item = "%s/%s %d.png" % (pt, fold, x1)
        cv2.imwrite(img_item, roi_color)
        color = (255, 0, 0)
        stroke = 2
        width = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        x1 = x1 + 1
    if x1 == 60:
        print("Done")
        break

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
