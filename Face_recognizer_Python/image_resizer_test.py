# Program that takes the images in one folder, resizes them using the basewodth
# and saves into another folder with the same name.
# Can be used for more efficient training(avoid overmathing) or for fisherfaces and eigenvectors
import PIL
from PIL import Image
import os

basewidth = 500
train_path = './train_set'
save_path = './scaled_train_set'
image_paths = [os.path.join(train_path, f)
    for f in os.listdir(train_path)]

for image_path in image_paths:
    img = Image.open(image_path)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,basewidth), PIL.Image.ANTIALIAS)
    # Extract filename out of path
    print image_path
    filename = image_path[image_path.find("train0")+0:].split()[0]
    print filename
    img.save('./scaled_train_set/' + filename,'JPEG')
