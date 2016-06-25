#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image
import sys

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

#LBPH face recognizer parameters:
RADIUS = 1
NEIGHBORS = 12
GRID_X = 9
GRID_Y = 9
THRESHOLD = 170.0
    
# For face recognition we will the the LBPH Face Recognizer
# Possibly use other 2 methods (fisherfaces and eigenvectors) for improved result
recognizer = cv2.createLBPHFaceRecognizer(RADIUS, NEIGHBORS, GRID_X, GRID_Y, THRESHOLD)


# Append all the absolute image paths in a list image_paths
def get_images_and_labels(train_set):
    # All images in folder are added to the image set
    image_paths = [os.path.join(train_set, f)
        for f in os.listdir(train_set)]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("train", ""))
        print nbr
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        # X and Y coordinates and width and height of the image
        #cv2.startWindowThread()
        #cv2.namedWindow("train_image")
        
        for (x, y, w, h) in faces: 
            images.append(image[y: y + h, x: x + w])# Add the height and width to the coordinates
            labels.append(nbr)
            
            #cv2.imshow("train_image", image[y: y + h, x: x + w])
            #cv2.waitKey(1)
    # return the images list and labels list
    return images, labels

# Path to the Dataset
train_set = './train_set'
test_set = './test_set'
# Call the get_images_and_labels function and get the face images and the 
# corresponding labels
images, labels = get_images_and_labels(train_set)
cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))

video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # One gray image from the feed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the face in this one image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(70, 70),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    # Create a new variables and if they are not empty, find who is it
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if (x and y and w and h) != 0:
            predict_image = np.array(gray, 'uint8')
            nbr_predicted, conf = recognizer.predict(gray)

            if conf < 50:
                print "{} is Correctly Recognized with confidence {}".format(nbr_predicted, conf)
            else:
                print "someone else"
        else:
            print "f"
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


