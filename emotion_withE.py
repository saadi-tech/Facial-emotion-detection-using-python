import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils




def check_eyes(roi):
    eye_detector = "haarcascade_eye.xml"
    detector = cv2.CascadeClassifier(eye_detector)
    rects = detector.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    return len(rects)


model_path= "checkpoints/epoch_90.hdf5"
cascade_path= "haarcascade_frontalface_default.xml"


# load the face detector cascade, emotion detection CNN, then define
# the list of emotion labels
detector = cv2.CascadeClassifier(cascade_path)
model = load_model(model_path)
EMOTIONS = ["angry", "scared", "happy", "sad", "surprised","neutral"]



cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resize the frame and convert it to grayscale
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # initialize the canvas for the visualization, then clone
    # the frame so we can draw on it
    canvas = np.zeros((220, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    # detect faces in the input frame, then clone the frame so that
    # we can draw on it
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    # ensure at least one face was found before continuing
    for i in range(0, len(rects)):
        # determine the largest face area
        # rect = sorted(rects, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = rects[i]
        # extract the face ROI from the image, then pre-process
        # it for the network
        roi = gray[fY:fY + fH, fX:fX + fW]

        eyes_found = 0
        eyes_found = check_eyes(roi)


        if(eyes_found>=1):
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            # make a prediction on the ROI, then lookup the class# label
            preds = model.predict(roi)[0]
            label = EMOTIONS[preds.argmax()]

            # loop over the labels + probabilities and draw them
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                # draw the label + probability bar on the canvas
                w = int(prob * 300)
                cv2.rectangle(canvas, (5, (i * 35) + 5), (w, (i * 35) + 35), (40, 50, 155), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 25, 5), 2)
            if(eyes_found>=2):
                cv2.putText(frameClone, "Focused and "+label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 50, 155), 2)
                print("Focused and ", label)
            else:
                cv2.putText(frameClone, "Slightly Focused and " + label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(40, 50, 155), 2)
                print("Slightly Focused and ", label)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (140, 50, 155), 2)
            # show our classifications + probabilities


        else:

            print("NOT FOCUSED...")

    if (len(rects) == 0):
        cv2.putText(frameClone, "NOT FOCUSED", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
        # cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (140, 50, 155), 2)
        print("NOT FOCUSED (no face found)")
    cv2.imshow("Face", frameClone)
    #cv2.imshow("Probabilities", canvas)
    #print("My cute ROMI is: ", label)
    # cleanup the camera and close any open windows
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()