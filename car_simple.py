# import the necessary packages
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
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
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()
	
#data["ActivityEncoded"] = le.fit_transform(data['activity'].values.ravel())

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

	if(i % 100 == 0):
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

	if(i % 100 == 0):
		print("[INFO] test data processed {}/{}".format(i, len(imagePaths)))
		
X_test = np.array(test_Images)
y_test = np.asarray(test_Labels)
	
	
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--learner", required=True,
	help="ML algorithm")
args = vars(ap.parse_args())


	
if(args["learner"] == "knn"):	
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=7, n_jobs=2)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("[INFO] kNN accuracy: {:.2f}%".format(acc * 100))


if(args["learner"] == "svm"):	
    from sklearn.svm import SVC
    model = SVC()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("[INFO] SVM accuracy: {:.2f}%".format(acc * 100))


if(args["learner"] == "dct"):	
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("[INFO] DT accuracy: {:.2f}%".format(acc * 100))


if(args["learner"] == "nb"):	
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("[INFO] NB accuracy: {:.2f}%".format(acc * 100))


    from sklearn.metrics import classification_report
    test_pred = model.predict(X_test)
    print('Classification Report of NB testing data:')
    print(classification_report(y_test, test_pred, digits=4, labels=CAR_LABELS))


if(args["learner"] == "rf"):	
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=800, random_state=12)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("[INFO] RF accuracy: {:.2f}%".format(acc * 100))


if(args["learner"] == "gbm"):	
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200, random_state=12)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("[INFO] GBM accuracy: {:.2f}%".format(acc * 100))


if(args["learner"] == "ann"):	
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(300,), max_iter=200, random_state=12)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("[INFO] MLP accuracy: {:.2f}%".format(acc * 100))


if(args["learner"] == "dl"):	
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(150,250,150,50), max_iter=300, random_state=12)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("[INFO] DL accuracy: {:.2f}%".format(acc * 100))





						
