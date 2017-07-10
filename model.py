import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# read data

filename = './record/driving_log.csv'
df = pd.read_csv(filenamexx, float_precision='high')
data = df.values
del df

# I select the three kinds of data

# In csv, the lines of data are as follow, 
# from 1 to 5117 loads counter_clockwise data 
# from 5118 to 7511 loads clockwise data
# from 7512 to 9294 loads the data from curve to middle
train_samples, validation_samples = train_test_split(data[:9294], test_size=0.2)
del data

# X_train.dtype = np.uint8; y_train.dtype = np.float64
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_img = cv2.imread(batch_sample[0])
                left_img = cv2.imread(batch_sample[1])
                right_img = cv2.imread(batch_sample[2])
                
                # cause the PIL Image method in drive.py read img as RGB, we need to train model with RGB
                center_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB)
                left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
                right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
                
                center_img_flipped = np.fliplr(center_img)
                
                images.extend([center_img, left_img, right_img, center_img_flipped])
                
                steering_center = float(batch_sample[3])
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                steering_center_flipped = -steering_center
                
                angles.extend([steering_center, steering_left, steering_right, steering_center_flipped])
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)  # actual we get 128 images in one call
validation_generator = generator(validation_samples, batch_size = 32)            
            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.models import load_model

# This is a regression network

# here we do two preprocessing
# (1) normalize the data
# (2) mean centering the data
def dave_2():
    input_shape = (160, 320, 3)

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((55, 25), (0, 0))))  # , input_shape=input_shape
    model.add(Conv2D(24, kernel_size=(5, 5),
                    activation='elu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(36, kernel_size=(5, 5),
                    activation='elu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(48, kernel_size=(5, 5),
                    activation='elu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, kernel_size=(3, 3),
                    activation='elu'))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(1164, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    print(model.summary())
    model.compile(loss='mse', optimizer='adam')

    steps_per_epoch = int(len(train_samples)/32)
    validation_steps =  int(len(validation_samples)/32)
    history_object = 0
    history_object = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=5, verbose=1, validation_data=validation_generator, validation_steps=validation_steps)
    model.save('model.h5')
    return history_object

history_object = dave_2()

K.clear_session()

# print the trainning and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()




