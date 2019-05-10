#!/usr/bin/python3
# -*- coding: utf-8 -*-

#MIT License
#
#Copyright (c) 2018 Justin Shenk
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import logging
# IMPORTANT:
#
# This code is derived from Iván de Paz Centeno's implentation of MTCNN
# (https://github.com/ipazc/mtcnn/) and Octavia Arriaga's facial expression recognition repo
# (https://github.com/oarriaga/face_classification).
#
import os
import sys

import cv2
import numpy as np
import pkg_resources
import tensorflow as tf
from keras.models import load_model

from fer.classes import Peltarion_Emotion_Classifier
from fer.exceptions import InvalidImage

__author__ = "Justin Shenk"

logging.basicConfig(
    format=
    '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO)


class FER(object):
    """
    Allows performing Facial Expression Recognition ->
        a) Detection of faces
        b) Detection of emotions
    """

    def __init__(self,
                 cascade_file: str = None,
                 mtcnn=False,
                 emotion_model: str = None,
                 scale_factor: float = 1.3,
                 min_face_size: int = 40,
                 offsets: tuple = (20, 40),
                 compile: bool = False):
        """
        Initializes the face detector and Keras model for facial expression recognition.
        :param cascade_file: file URI with the Haar cascade for face classification
        :param mtcnn: use MTCNN network for face detection (not yet implemented)
        :param emotion_model: file URI with the Keras hdf5 model or Peltarion API URL
        :param scale_factor: parameter specifying how much the image size is reduced at each image scale
        :param min_face_size: minimum size of the face to detect
        :param offsets: padding around face before classification
        :param compile: value for Keras `compile` argument
        """
        self.__scale_factor = scale_factor
        self.__min_face_size = min_face_size
        self.__offsets = offsets

        if cascade_file is None:
            cascade_file = pkg_resources.resource_filename(
                'fer', 'data/haarcascade_frontalface_default.xml')

        if mtcnn:
            self.__face_detector = 'mtcnn'
            raise NotImplementedError("To be implemented")
        else:
            self.__face_detector = cv2.CascadeClassifier(cascade_file)

        if not emotion_model:
            # Local Keras model
            self.deployment = False
            emotion_model = pkg_resources.resource_filename(
                'fer', 'data/emotion_model.hdf5')
            config = tf.ConfigProto(log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.__emotion_classifier = load_model(
                emotion_model, compile=compile)
            self.__emotion_classifier._make_predict_function()
            self.__emotion_target_size = self.__emotion_classifier.input_shape[
                1:3]
        elif 'http' in emotion_model:
            self.deployment = True
            url = os.environ.get('EMOTION_API_URL')
            token = os.environ.get('EMOTION_API_TOKEN')
            assert url is not None and token is not None, "EMOTION_API_URL and EMOTION_API_URL must set in the environment"
            self.__emotion_classifier = Peltarion_Emotion_Classifier(
                url, token)
            self.__emotion_target_size = (48, 48)  # Default FER image size
        else:
            raise Exception(f"{emotion_model} is not a valid type")
        logging.debug("Emotion model: {}".format(emotion_model))

    @property
    def min_face_size(self):
        return self.__min_face_size

    @min_face_size.setter
    def min_face_size(self, mfc=50):
        try:
            self.__min_face_size = int(mfc)
        except ValueError:
            self.__min_face_size = 50

    @staticmethod
    def pad(image):
        row, col = image.shape[:2]
        bottom = image[row - 2:row, 0:col]
        mean = cv2.mean(bottom)[0]

        bordersize = 40
        padded_image = cv2.copyMakeBorder(
            image,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[mean, mean, mean])
        return padded_image

    @staticmethod
    def depad(image):
        row, col = image.shape[:2]
        return image[40:row - 40, 40:col - 40]

    def find_faces(self, gray_image_array):
        faces = self.__face_detector.detectMultiScale(
            gray_image_array,
            scaleFactor=self.__scale_factor,
            minNeighbors=5,
            flags=0,
            minSize=(self.__min_face_size, self.__min_face_size))
        return faces

    @staticmethod
    def __preprocess_input(x, v2=True):
        x = x.astype('float32')
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x

    def __apply_offsets(self, face_coordinates):
        x, y, width, height = face_coordinates
        x_off, y_off = self.__offsets
        return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

    @staticmethod
    def _get_labels():
        return {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'sad',
            5: 'surprise',
            6: 'neutral'
        }

    def detect_emotions(self, img) -> list:
        """
        Detects bounding boxes from the specified image with ranking of emotions.
        :param img: image to process
        :return: list containing all the bounding boxes detected with their emotions.
        """
        if img is None or not hasattr(img, "shape"):
            raise InvalidImage("Image not valid.")

        emotion_labels = self._get_labels()

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_rectangles = self.find_faces(gray_img)
        emotions = []
        for face_coordinates in face_rectangles:
            x1, x2, y1, y2 = self.__apply_offsets(face_coordinates)
            gray_face = gray_img[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, self.__emotion_target_size)
            except Exception as e:
                print("{} resize failed".format(gray_face.shape))
                continue
            if not self.deployment:
                # Local Keras model
                gray_face = self.__preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = self.__emotion_classifier.predict(
                    gray_face)[0]
                labelled_emotions = {
                    emotion_labels[idx]: round(score, 2)
                    for idx, score in enumerate(emotion_prediction)
                }
            elif self.deployment:
                # Peltarion API
                emotion_prediction = self.__emotion_classifier.predict(
                    gray_face)
                labelled_emotions = {
                    emotion: round(score, 2)
                    for emotion, score in emotion_prediction.items()
                }
            else:
                raise NotImplemented()

            emotions.append({
                'box': face_coordinates,
                'emotions': labelled_emotions
            })
        return emotions


def parse_arguments(args):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Image filepath')
    return parser.parse_args()


def inference():
    args = parse_arguments(sys.argv)
    fer = FER()
    inference = fer.detect_emotion(args.image)
    print(inference)


def main(args=None):
    pass


if __name__ == '__main__':
    main()
