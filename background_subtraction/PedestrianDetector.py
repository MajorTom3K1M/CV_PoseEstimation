from imutils.object_detection import non_max_suppression
import cv2 as cv
import imutils
import numpy as np

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

cap = cv.VideoCapture('vid.mp4')

while(True):
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=min(400, frame.shape[1]))
    orig = frame.copy()

    # frame = cv.GaussianBlur(frame, (3, 3), 0)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    (rects, weights) = hog.detectMultiScale(frame, winStride=(2, 2),
        padding=(4, 4), scale=1.05)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in rects:
        cv.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    frame = imutils.resize(frame, width=400)
    cv.imshow("Video", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
