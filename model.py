import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import math
import csv
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D, Dropout, ELU, Lambda
from keras.models import Sequential
import cv2


data=[]

def load_file(csv_path):
    steering=0
    with open(csv_path,"r") as f:
        reader = csv.reader(f, delimiter = ",")
        for row in reader:
            if row[0] == 'center': continue
            line = {}
            line['center'],line['left'],line['right'] = row[0],row[1],row[2]
            line['prev_steering'] = steering
            steering = float(row[3])
            line['steering'] = steering
            line['speed'] = float(row[6])
            data.append(line)

def get_img(file):
    return cv2.imread("./data/" + file.strip())

def augmentation_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = img_shape[0]
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def augmentation_trans(image, steer, trans_range):
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (img_shape[1], img_shape[0]))

    return image_tr, steer_ang

def augmentation_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = np.random.uniform(0.25, 1.25)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def augmentation_crop(img, steering):
    #img[0: math.floor(img_shape[0] / 5), :, :] = int(steering*128+127)
    #img[img_shape[0] - 25: img_shape[0], :, :] = int(steering*128+127)

    img = img[math.floor(img_shape[0] / 5):img_shape[0] - 25, :, :]
    return img

def augmentation_flip(img, steering):
    i = np.random.randint(2)
    if i == 0:
        img = cv2.flip(img, 1)
        steering = - steering
    return img, steering

def augmentation_select_camera(line):
    camera = np.random.choice(['center', 'left', 'right'])
    steering = line["steering"]
    if camera == 'left':
        steering += 0.25
    elif camera == 'right':
        steering -= 0.25

    return get_img(line[camera]), steering

def preprocess_line(line):
    image, steering = augmentation_select_camera(line)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = augmentation_brightness(image)
    image, steering = augmentation_trans(image, steering, 100)
    image = augmentation_crop(image, line["prev_steering"])

    image = augmentation_shadow(image)

    image = np.array(image)
    image, steering = augmentation_flip(image, steering)

    return image, steering

def gen(batch_size): #generator
    while 1:
        X = np.zeros((batch_size, crop_shape[0], crop_shape[1], crop_shape[2]))
        Y = np.zeros(batch_size)
        for i in range(batch_size):
            row = np.random.randint(csv_count)
            x, y = preprocess_line(data[row])
            X[i] = x
            Y[i] = y

        yield X,Y

def get_model_commaai(): #source of the model https://github.com/commaai/research/blob/master/train_steering_model.py
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=crop_shape, output_shape=crop_shape))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    model.summary()

    return model

def get_model_nvidia(): #source of the model: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
  model = Sequential()
  model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=crop_shape, output_shape=crop_shape))
  model.add(Convolution2D(3, 5, 5,  subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(48, 3, 3, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
  model.add(ELU())

  model.add(Flatten())
  model.add(Dropout(.2))

  model.add(Dense(1164))
  model.add(Dropout(.2))
  model.add(ELU())

  model.add(Dense(100))
  model.add(Dropout(.2))
  model.add(ELU())

  model.add(Dense(50))
  model.add(Dropout(.5))
  model.add(ELU())

  model.add(Dense(10))
  model.add(Dropout(.5))
  model.add(ELU())

  model.add(Dense(1))

  model.compile(optimizer="adam", loss="mse")

  model.summary()

  return model



#######################################################################################################################

# 0. load data
load_file('./data/driving_log.csv')
csv_count = len(data)

img_shape = get_img(data[0]['center']).shape
print(img_shape)
crop_shape = augmentation_crop(get_img(data[0]['center']),0).shape
print(crop_shape)

# test augmentation
# i = 0
# for x, y in gen(1):
#     plt.imsave('./test/'+str(i)+'.jpg', x[0])
#     i+=1
#     if i>=100: break;

# 1. define hyperparameters
batch_size = 512
samples_per_epoch = 40000
epochs = 10
val_samples = 3000

# 2. define generators
gen_train = gen(batch_size)
gen_validate = gen(batch_size)

# 3. create model model
#model = get_model_commaai()
model = get_model_nvidia()

# 4. train and save
history = model.fit_generator(gen_train, samples_per_epoch = samples_per_epoch//batch_size*batch_size, nb_epoch = epochs,
                              verbose=1, validation_data=gen_validate, nb_val_samples=val_samples//batch_size*batch_size)

import json
with open('./model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
model.save_weights('model.h5')

# 5. check
val = []
val.append(augmentation_crop(get_img('IMG/center_2016_12_01_13_31_15_208.jpg'), 0)) # 0
val.append(augmentation_crop(get_img('IMG/center_2016_12_01_13_45_13_420.jpg'), 0)) # -0.4110484
val.append(augmentation_crop(get_img('IMG/center_2016_12_01_13_45_07_636.jpg'), 0)) # 0.4540697

prediction = model.predict(np.array(val))
print(prediction)
