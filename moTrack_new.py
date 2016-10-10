import imutils
import time
import cv2

previousFrame = None

def searchForMovement(cnts, frame, min_area):
    
    text = "Undetected"
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = "Detected"

    return frame, text

def trackMotion(camera, min_area):

    ret, frame = camera.read()

    # Convert to grayscale and blur it for better frame difference
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (51, 51), 0)
    
    global previousFrame
    if previousFrame is None:
        previousFrame = gray
        return frame, "Uninitialized", frame, frame

    frameDiff = cv2.absdiff(previousFrame, gray)
    thresh = cv2.threshold(frameDiff, 25, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    frame, text = searchForMovement(cnts, frame, min_area)

    return frame, text, thresh, frameDiff

if __name__ == '__main__':
    camera = cv2.VideoCapture(1)
    time.sleep(0.25)
    min_area = 5000 #int(sys.argv[1])
    while True:
        frame, text, thresh, frameDiff = trackMotion(camera, min_area)
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Security Camera Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Difference", frameDiff)
        key = cv2.waitKey(3) & 0xFF
        if key == 27 or key == ord('q'):
            print("bye")
            break
    
    camera.release()
    cv2.destroyAllWindows()

