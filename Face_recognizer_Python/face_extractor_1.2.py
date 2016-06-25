# Converts images to grayscale, finds face in the image and crops the rest
# saves the new image into another folder

import cv2 as cv #Opencv
import PIL
from PIL import Image
import glob
import os
import numpy as np

cascadePath = "haarcascade_frontalface_alt.xml"
faceCascade = cv.CascadeClassifier(cascadePath)

destination_path = "./train_set/"
origin_path = "./scaled_train_set/"
image_name = "test0"

print origin_path
# Size of the resulting box from the image
basewidth = 300

def DetectFace(image, faceCascade, returnImage=False):

    # Equalize the histogram
    cv.equalizeHist(image, image)

    # Detect the faces
    faces = faceCascade.detectMultiScale(image)
    # Use exeption to handle the case when no face has been found and size of array is 0
    try:
        face_exists=faces.any()
    except:
        face_exists = 0
    if face_exists != 0:
        for (x, y, w, h) in faces:
            # Convert bounding box to two CvPoints
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv.rectangle(image, pt1, pt2, (255, 0, 0), 5, 8, 0)
        
    if returnImage:
        return image
    else:
        return faces

def cv2pil(cv_im):
    # Convert the cv image to a PIL image
    return Image.fromstring("L", cv.GetSize(cv_im), cv_im.tostring())

def imgCrop(image, cropBox, boxScale=1):
    # Crop a PIL image with the provided box [x(left), y(upper), w(width), h(height)]

    # Calculate scale factors
    xDelta=max(cropBox[2]*(boxScale-1),0)
    yDelta=max(cropBox[3]*(boxScale-1),0)

    # Convert cv box to PIL box [left, upper, right, lower]
    PIL_box=[cropBox[0]-xDelta, cropBox[1]-yDelta, cropBox[0]+cropBox[2]+xDelta, cropBox[1]+cropBox[3]+yDelta]

    return image.crop(PIL_box)

def faceCrop(path,boxScale=1):
    
    faceCascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

    image_paths = [os.path.join(path, f)
        for f in os.listdir(path)]

    for img in image_paths:
        pil_im=Image.open(img).convert('L')
        image = np.array(pil_im, 'uint8')
        faces=DetectFace(image,faceCascade)
        try:
            faces_size = faces.any
        except:
            faces_size = 0
            print "no face found on {}".format(img)
        if faces_size != 0:
            n=1
            for face in faces:
                croppedImage=imgCrop(pil_im, face,boxScale=boxScale)
                croppedImage = croppedImage.resize((basewidth,basewidth), PIL.Image.ANTIALIAS)
                fname,ext=os.path.splitext(img)
                filename = img[img.find(image_name)+0:].split()[0]
                print filename
                croppedImage.save(destination_path + filename,'JPEG')
                n+=1


# Prompts user to enter relative path to the images in the same folder as the program
# e.g. "./scaled_train_set/" is the correct form to enter destination path
origin_path = raw_input("Enter folder where images are located:")
destination_path = raw_input("Enter folder where to save new images:")
image_name = raw_input("Enter image name e.g. train0 or test0:")

faceCrop(origin_path, boxScale=1)
