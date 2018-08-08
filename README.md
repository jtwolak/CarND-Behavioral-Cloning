# Self-Driving Car Engineer Nanodegree

------

**Term1: Behavioral-Cloning - Project 3**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/placeholder.png "Model Visualization"

### 

###Project Files 

####1. Files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md  summarizing the results

####2. Running
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Source code

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. Model architecture

My initial approach was to retrain my LeNet based model, I used for the sign classification project, but after a few failed training attempts to keep the car on track I decided to look for a more advanced model. The model developed by Nvidia Autopilot group gained some publicity in community recently so I decided to give it a try. The original Nvidia-Autopilot model I used can be found here:

https://github.com/0bserver07/Nvidia-Autopilot-Keras/blob/master/model.py

The model includes five RELU layers to introduce nonlinearity (code lines 129-133), and the data is normalized in the model using a Keras lambda layer on line # 127. 

####2. How to reduce overfitting in the model

I decided to try the model without optimizations first.  To my surprise after collecting sufficient amount of training data the model performed very well "out-of-the-box". The overfitting problem, initially observed with very small data set, was solved by collecting more data driving the car clock wise and  counter clock wise and also using the data from the left and right cameras. 

####3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 147).

####4. Training data

Training data was collected by driving the car in the center of the road only. But I also used the images from the lest ad right cameras to make the data set more general.

For details about how I created the training data, see the next section. 

###Model Architecture Tuning and Re-Training Strategy

####1. Model Architecture Tuning

My first step was to use a very simple single layer convolutional model and see how the car would behave ... and it went straight to the lake. My next attempt was to try extending the LeNet model I used for the sign detection project by addition of dropout layers ( lines 109, 112 i 115) to help the model to generalize better. I also added the normalization layer (Line 107) to help the optimizer. I had some success keeping the car on the track, but it was clear that it will be very difficult to make the full circle. 

So on the third and final attempt I tried  to train the NVIDIA Autopilot model. My first attempt to train the model was to just use the data collected from one lap from the center camera only.  The first run was already very encouraging, because the car made it to about half the track, however it was also clear that it is not enough to make the full circle.  

After trying to increase the number of epochs with no success I realized, that the model was overfitting, because it had low mean squared error on the training set, but a high mean squared error on the validation set. To solve this problem I decided to collect more data. As the second step I flipped the data I already had vertically and I re-trained the model. This time the car made almost a full circle, but it ended up in the water again. As the third step I decided to use the data from the left and right camera. With the data from the right and left camera the car made the whole circle, but occasionally touched the curb. As the fourth and final step I collected the data driving counter clock wise. With this additional data  the model was able to  make the full lap without any problems.  



####2. Final Model Architecture

The model architecture (model.py lines 125-139) consisted of a convolution neural network with the following layers and layer sizes visualized in the diagram below:

- ![alt text][image1]


####3. Creation of the Training Set & Training Process

To generate more data I performed the following steps:

1. I captured data from center camera data driving counter clock wise - 3864 images
2. I flipped the data vertically generating additional - 3864 images
3. I added the images from left and right camera - 7728 images
4. I captured data from center camera driving clock wise - 15184 images
5. TOTAL:  30640 samples

After the collection process, I had 30640 samples of data.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7.  I used an dam optimizer so that manually training the learning rate wasn't necessary.
