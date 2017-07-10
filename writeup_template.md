# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_image.jpg "center image"
[image3]: ./examples/recovery/1.jpg "Recovery Image"
[image4]: ./examples/recovery/2.jpg "Recovery Image"
[image5]: ./examples/recovery/3.jpg "Recovery Image"
[image6]: ./examples/recovery/4.jpg "Recovery Imagee"
[image7]: ./examples/recovery/5.jpg "Recovery Image"
[image8]: ./examples/recovery/6.jpg "Recovery Image"
[image9]: ./examples/model_architecture.png "Model Architecture"
[image10]: ./examples/fell_off_track.png "fell off track"
[image11]: ./examples/fork_fail.png "fork fail"
[image12]: ./examples/memory_error.png "memory error"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is built according to [end to end self-driving paper wrote by Nvidia](https://arxiv.org/pdf/1604.07316.pdf).

which consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 81-92) that includes ELU layers to introduce nonlinearity each except the last conv layer.

The data is normalized in the model using a Keras lambda layer (code line 79) and trim range between row 55 to 135 each image to get rid of noisy information such as beautiful scenery and the car hood. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers with rate = 0.5 in order to reduce overfitting (model.py line 95). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 22). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 102).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

Data distribution as follow:

| Probability  |         Data Type         |                Prediction                |
| :----------: | :-----------------------: | :--------------------------------------: |
|  1 to 5117   |     counter-clockwise     | with 5117 for each, totally 5117*4=20468 (center, left, right, horizontally flip according to center) |
| 5118 to 7511 |         clockwise         |     with 2394 for each, totally 9576     |
| 7512 to 9293 | data from curve to center |     with 1782 for each, totally 7128     |

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to My architecture used in traffic sign classification.

At first, I tried to use the model with five convolutional layer whose filter number are 32, 64, 64, 128, 128 in sequence and two fully connected layer. However, after reading the paper [End-to-End Self-driving](https://arxiv.org/pdf/1604.07316.pdf), I was astounded by such a small model that could fit better than my original model.

The overall strategy for deriving a model architecture was from the paper mentioned above.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

To prevent the overfitting, I modified the model by adding Dropout Layer right after the first fully connected layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like the picture below. to improve the driving behavior in these cases, I add more data from edge to curve.

![alt text][image10]



At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

|      Layer       |               Description                |
| :--------------: | :--------------------------------------: |
|      Input       |            32x32x3 RGB image             |
|   Lambda Layer   | data normalize with (pixel / 255.) - 0.5 |
| Cropping2D Layer | trim image to get rid of the information we don't need. |
| Convolution 5x5  | 1x1 stride, same padding, outputs 76x316x24 |
|       ELU        |                                          |
|   Max pooling    |      2x2 stride,  outputs 38x158x24      |
|                  |                                          |
| Convolution 5x5  | 1x1 stride, valid padding, outputs 34x154x36 |
|       ELU        |                                          |
|   Max pooling    |      2x2 stride,  outputs 17x77x36       |
|                  |                                          |
| Convolution 5x5  | 1x1 stride, valid padding, outputs 13x73x48 |
|       ELU        |                                          |
|   Max pooling    |       2x2 stride,  outputs 6x36x48       |
|                  |                                          |
| Convolution 3x3  | 1x1 stride, valid padding, outputs 4x34x64 |
|       ELU        |                                          |
| Convolution 3x3  | 1x1 stride, valid padding, outputs 2x32x64 |
|       ELU        |                                          |
|                  |                                          |
| Fully connected  |        inputs 4096, outputs 1164         |
|     Dropout      |                rate = 0.5                |
| Fully connected  |               outputs 100                |
| Fully connected  |                outputs 50                |
| Fully connected  |                outputs 10                |
| Fully connected  | outputs 1 (Regression Model no need softmax) |


![alt text][image9]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one with counter clock-clockwise using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

Second I recorded two laps on track one with clock-clockwise using center lane driving.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from edge to center. These images show what a recovery looks like starting from edge to center: (just the concept, in reality, the sample rate is high that the change below may generate 16 images)

![alt text][image4]![alt text][image3]

![alt text][image5] ![alt text][image6]

![alt text][image7] ![alt text][image8]	



To augment the data set, I also horizontally flipped images and angles of center image when running the model training. I thinking that this can double the dataset of center image of counter-clockwise and clockwise.

After the collection process, I had 36,772 number of data points. I then preprocessed this data by using Cropping2D and Lambda Layer provided by Keras.

**Note.** I found the important fact that if I use the Lambda layer in my Keras model which train under Ubuntu 16.04 with tensorflow gpu backend, that model coundn't be loaded by my Macbook which use cpu tensorflow backend.

The generator function and the Lambda layer in Keras are so important that my computer would get memory error without them. (36,772 images with dtype np.float32 exhaust my 32 GB ram.)

![alt text][image12]

![alt text][image11]

The generator function dynamically load image into memory with number of batch_size. (code line 26-59 in model.py). And the images we loaded are dtype uint8.

It is the Lambda layer takes uint8 image, do data normalize and output the image with np.float32 which dramatically reduces the memory consumption. (code line 79 in model.py)


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the difference between train loss and validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
