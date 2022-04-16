import cv2

cascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml') # Haar Cascade classifier which is pretrained XML file

cap = cv2.VideoCapture(0) # for capturing video
cap.set(3,2000)
cap.set(4,3000) # setting windows width and height

while True:
    success, img = cap.read() # Reading images frame by frame
    face = cascade.detectMultiScale(img, 1.1, 4)  # detect multiple face

    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),3)  # bounding box width and height
    cv2.imshow('output', img)
    if cv2.waitKey(10) & 0xFF == ord(' '):  #stopping from automatic closing of window
        break

## Let's run and see what happens

## As you can see it is not precise and predicting false faces also (Not Robust)

## lets move to other approach