#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image

# Path to the faces collection
train_path = './scaled_train_set'
test_path = './test_set'
update_path = './update_set' # @todo

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

#LBPH face recognizer parameters:
RADIUS = 1
NEIGHBORS = 14
GRID_X = 9
GRID_Y = 9
THRESHOLD = 170.0
    
# For face recognition we will the the LBPH Face Recognizer
# Possibly use other 2 methods (fisherfaces and eigenvectors) for improved result
recognizer_LBPH = cv2.createLBPHFaceRecognizer(RADIUS, NEIGHBORS, GRID_X, GRID_Y, THRESHOLD)
recognizer_Fisher = cv2.createFisherFaceRecognizer()

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    # Add image path to the image_paths if its extension is not .sad
    size = 100,100
    image_paths = [os.path.join(path, f)
        for f in os.listdir(path)]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        #image_pil.thumbnail(size, Image.ANTIALIAS)
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("train", ""))
        print image_path
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        print faces
        # If face is detected, append the face to images and the label to labels
        # X and Y coordinates and width and height of the image
        cv2.startWindowThread()
        cv2.namedWindow("training model")
        for (x, y, w, h) in faces: 
            images.append(image[y: y + h, x: x + w])# Add the height and width to the coordinates
            labels.append(nbr)
            cv2.imshow("training model", image[y: y + h, x: x + w])
            cv2.waitKey(10)
    # return the images list and labels list
    return images, labels

def get_prediction():
    # Load the available model:
    recognizer_LBPH.load("LBPH_face_train_model")
    # recognizer_Fisher.load("Fisher_face_train_model")
    # Append the images with the extension .sad into image_paths
    image_paths = [os.path.join(test_path, f)
        for f in os.listdir(test_path)]
    for image_path in image_paths:
        print image_path
        predict_image_pil = Image.open(image_path).convert('L')
        predict_image = np.array(predict_image_pil, 'uint8')
        faces = faceCascade.detectMultiScale(predict_image)
        #print faces
        cv2.startWindowThread()
        cv2.namedWindow("predictions..")
        for (x, y, w, h) in faces:
            LBPH_nbr_predicted, LBPH_conf = recognizer_LBPH.predict(predict_image[y: y + h, x: x + w])
            # Fisher_nbr_predicted, Fisher_conf = recognizer_Fisher.predict(predict_image[y: y + h, x: x + w])
            # print Fisher_nbr_predicted, Fisher_conf
            nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("test", ""))
            if nbr_actual == LBPH_nbr_predicted:
                print "{} is Correctly Recognized with confidence {}".format(nbr_actual, LBPH_conf)
            else:
                print "{} is Incorrect Recognized as {} with confidence {}".format(nbr_actual, LBPH_nbr_predicted, LBPH_conf)
            cv2.imshow("predictions..", predict_image[y: y + h, x: x + w])
            cv2.waitKey(100)
            
var = raw_input("Press N to create the model and U to update it: ")
if var == "N":
    # Call the get_images_and_labels function and get the face images and the 
    # corresponding labels
    images, labels = get_images_and_labels(train_path)
    cv2.destroyAllWindows()

    # Perform the tranining
    recognizer_LBPH.train(images, np.array(labels))
    # recognizer_Fisher.train(images, np.array(labels))
    
    # Save the model to save time
    recognizer_LBPH.save("LBPH_face_train_model")
    # recognizer_Fisher.save("Fisher_face_train_model")
    print "Model trained"
elif var == "U":
    # Add extra images to the model
    images, labels = get_images_and_labels(update_path)
    cv2.destroyAllWindows()

    #Load the available model:
    recognizer_LBPH.load("face_train_model")
    
    # Perform the tranining
    recognizer_LBPH.train(images, np.array(labels))
    
    recognizer.update(images, np.array(labels))
    # Save the model to save time
    print "Model updated"
else:
    get_prediction()



