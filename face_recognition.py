import cv2
import pickle

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
cap = cv2.VideoCapture(0)
x1 = 0

labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # print(x,y,w,h)
        roi_gray = gray[y:y + h, x:x + h]
        roi_color = frame[y:y + h, x:x + h]
        cv2.imshow('frame2', roi_gray)
        id_, conf = recognizer.predict(roi_gray)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 0)
        stroke = 2
        if conf > 50:
            print(id_)
            name = labels[id_]
            color1 = (255, 255, 255)
            cv2.putText(frame, name, (x, y), font, 1, color1, stroke, cv2.LINE_AA)
        else:
            name = "Unknown"
            color1 = (255, 255, 255)
            cv2.putText(frame, name, (x, y), font, 1, color1, stroke, cv2.LINE_AA)

        width = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
