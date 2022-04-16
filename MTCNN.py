import mtcnn
from mtcnn.mtcnn import MTCNN
import cv2


detector = MTCNN()  # MTCNN is CNN based algorithm

video = cv2.VideoCapture(0)
video.set(3,2000)
video.set(4,3000) # Same as previous technique

while (True):
    ret, frame = video.read()
    if ret == True:
        location = detector.detect_faces(frame)  # dectect faces frame by frame
        if len(location) > 0:
            for face in location:
                x, y, width, height = face['box']
                x2, y2 = x + width, y + height
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 4) # Bounding box width and height
        cv2.imshow("Output",frame) # its will show frame's and update it frame by frame to same output file named as "Output"
        if cv2.waitKey(1) & 0xFF == ord(' '):   # same as previous
            break
    else:
        break

video.release()  # releasing camera port
cv2.destroyAllWindows() # destroying all windows



# Lets run and see , Results
# as you can see, it is giving precise result

# so we will go with MTCNN Algorithm

