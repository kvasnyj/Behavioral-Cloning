import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import math
import cv2

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf


def crop(img, steering):
    img_shape = img.shape
    #img[0: math.floor(img_shape[0] / 5), :, :] = int(float(steering)/30*128+127)
    #img[img_shape[0] - 25: img_shape[0], :, :] = int(float(steering)/30*128+127)

    img = img[math.floor(img_shape[0] / 5):img_shape[0] - 25, :, :]
    return img

tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car

    imgString = data["image"]

    image = Image.open(BytesIO(base64.b64decode(imgString)))

    image_array = np.asarray(image)

    image_array = crop(image_array, steering_angle)

    transformed_image_array = image_array[None, :, :, :]

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    angle =  model.predict(transformed_image_array, batch_size=1)
    steering_angle = float(angle)

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    brake = 0
    if float(speed) < 10:
        throttle = 0.3
    elif float(speed)<20:
        throttle = 0.2
    else:
        throttle = 0.0 # only for descent after tunnel on track 2
        brake= (speed-20.)/10. #not working :(


    print(steering_angle, throttle, brake)
    send_control(steering_angle, throttle, brake)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0, 0)


def send_control(steering_angle, throttle, brake):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__(),
    'brake': brake.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        #model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)