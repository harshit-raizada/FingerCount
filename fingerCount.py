import mediapipe as mp
import cv2 as cv

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
fingerCords = [(8,6), (12,10), (16,14), (20,18)]
thumbCords = (4,2)

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        handList = []
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                handList.append((cx, cy))
    
        count = 0
        for coordinates in fingerCords:
            if handList[coordinates[0]][1] < handList[coordinates[1]][1]:
                count = count + 1
        if handList[thumbCords[0]][0] > handList[thumbCords[1]][0]:
            count = count + 1

        cv.putText(img, f'Count: {str(count)}', (50, 100), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 4)
    
    cv.imshow('Image', img)
    k = cv.waitKey(1)

    if k == ord('q'):
        break