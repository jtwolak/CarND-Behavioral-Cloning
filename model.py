import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Lambda, Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Load the images and the steering measurements from the data collected from the simulator
def load_images( dataset_path, images = [], measurements = [] ):

    lines = []
    csv_file_name = dataset_path + '/driving_log.csv'
    with open(csv_file_name) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines:
        source_path = line[0]
        if source_path[0] == 'C':
            filename = source_path.split('\\')[-1]
        if source_path[0] != "C":
            filename = source_path.split('/')[-1]
        if filename == "center":
            continue
        current_path = dataset_path + '/IMG/' + filename
        inputImage = cv2.imread(current_path)
        image = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
        images.append(image)

        # Append the center image
        measurement = float(line[3])
        measurements.append(measurement)

        # Append the flipped image with angle adjustment
        images.append(cv2.flip(image,1))
        measurements.append(measurement*-1.0)

        # Append the left image with "+" correction
        source_path = line[1]
        if source_path[0] == 'C':
            filename = source_path.split('\\')[-1]
        if source_path[0] != "C":
            filename = source_path.split('/')[-1]
        current_path = dataset_path + '/IMG/' + filename
        inputImage = cv2.imread(current_path)
        image = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
        images.append(image)
        measurements.append(measurement+0.2)

        # Append the right image with "-" correction
        source_path = line[2]
        if source_path[0] == 'C':
            filename = source_path.split('\\')[-1]
        if source_path[0] != "C":
            filename = source_path.split('/')[-1]
        current_path = dataset_path + '/IMG/' + filename
        inputImage = cv2.imread(current_path)
        image = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
        images.append(image)
        measurements.append(measurement-0.2)


images = []
measurements = []
#load_images( "data", images, measurements )
#load_images( "C:/tmp/data", images, measurements )
load_images( "C:/Udacity/data 1", images, measurements )
load_images( "C:/Udacity/data 3", images, measurements )

# Now that I loaded images and the steering measurements I am going to convert them to numpy arrays
# because that is the format Keras requires.
X_train = np.array(images)
y_train = np.array(measurements)

# Next I am going to build the basic neural network possible, just to verify that everything is working.
# This model is going to be a flattened image connected to a single output node. This single output node will
# predict my steering angle which makes it a regression network.
# For classification network I might apply softmax activation function to the output layer. But in the regression
# network like this I just what the output node to predict the steering measurement so I won't apply and activation
# function here.

# 1. Start model
def basicModel():
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    return model

# 2. Add normalization
def basicModelNorm():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Flatten())
    model.add(Dense(1))
    return model

# 3. Use LeNet model
def LeNetNorm():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Dropout(0.5))
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

# 4. use Nvidia autopilot model
# This is my final model.

def nVidiaAutopilotNorm():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

model = nVidiaAutopilotNorm()

# With the network constructed I compile the model. For the loss function I use mean-square-error or MSE,
# This is different then the cross-entropy function we have used in the past,because this a regression network
# instead of classification network. What I want is to minimize the error between the steering measurement that the
# network predicts and the ground truth measurement. MSE is a good loss function for this.
model.compile(loss='mse',optimizer='adam')

# Once the model is compiled I';ll train it with the feature and label arrays I have just built. I'll also shaffle
# the data and split 20% of the data to use for validation set.
#model.fit(X_train,y_train, validation_split=0.2, shuffle=True)
result = model.fit( X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)
print(result.history.keys())
print('Loss')
print(result.history['loss'])
print('Validation Loss')
print(result.history['val_loss'])

# Finally I am going to save the trained models so that later I can download it to my local machine and see it it
# works for driving in teh simulator
model.save('model.h5')









