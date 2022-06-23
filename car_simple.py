# import the necessary packages
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import imutils
import cv2
import os
from sklearn import preprocessing

CAR_LABELS = [
    'Audi', 
    'Hyundai Creta', 
    'Mahindra Scorpio', 
    'Rolls Royce', 
    'Swift', 
    'Tata Safari', 
    'Toyota Innova']

le = preprocessing.LabelEncoder()

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image
	return cv2.resize(image, size).flatten()
	
# loop over the training images
train_Images = []
train_Labels = []
imagePaths = list(paths.list_images("images/train"))
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = os.path.basename(os.path.dirname(imagePath))

    pixels = image_to_feature_vector(image)

    train_Images.append(pixels)
    train_Labels.append(label)

    print("[INFO] train data processed {}/{}".format(i, len(imagePaths)))

X_train = np.array(train_Images)
y_train = np.asarray(train_Labels)

# loop over the testing images
test_Images = []
test_Labels = []
imagePaths = list(paths.list_images("images/test"))
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = os.path.basename(os.path.dirname(imagePath))

    pixels = image_to_feature_vector(image)

    test_Images.append(pixels)
    test_Labels.append(label)

    print("[INFO] test data processed {}/{}".format(i, len(imagePaths)))
		
X_test = np.array(test_Images)
y_test = np.asarray(test_Labels)

from sklearn.linear_model import SGDClassifier
model = SGDClassifier(random_state=12)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print("[INFO] SGD accuracy: {:.2f}%".format(acc * 100))
