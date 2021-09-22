#!/usr/bin/python3
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2018 Justin Shenk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# IMPORTANT:
#
# This code is derived from Iván de Paz Centeno's implementation of MTCNN
# (https://github.com/ipazc/mtcnn/) and Octavia Arriaga's facial expression recognition repo
# (https://github.com/oarriaga/face_classification).
#
import sys

import cv2
import numpy as np
import pkg_resources

from typing import Sequence, Tuple, Union

from tensorflow.keras.models import load_model

from .exceptions import InvalidImage
from .logger import log

NumpyRects = Union[np.ndarray, Sequence[Tuple[int, int, int, int]]]

__author__ = "Justin Shenk"


class FER(object):
    """
    Allows performing Facial Expression Recognition ->
        a) Detection of faces
        b) Detection of emotions
    """

    def __init__(
        self,
        cascade_file: str = None,
        mtcnn=False,
        emotion_model: str = None,
        scale_factor: float = 1.1,
        min_face_size: int = 50,
        min_neighbors: int = 5,
        offsets: tuple = (10, 10),
        compile: bool = False,
    ):
        """
        Initializes the face detector and Keras model for facial expression recognition.
        :param cascade_file: file URI with the Haar cascade for face classification
        :param mtcnn: use MTCNN network for face detection (not yet implemented)
        :param emotion_model: file URI with the Keras hdf5 model
        :param scale_factor: parameter specifying how much the image size is reduced at each image scale
        :param min_face_size: minimum size of the face to detect
        :param offsets: padding around face before classification
        :param compile: value for Keras `compile` argument
        """
        self.__scale_factor = scale_factor
        self.__min_neighbors = min_neighbors
        self.__min_face_size = min_face_size
        self.__offsets = offsets

        if cascade_file is None:
            cascade_file = pkg_resources.resource_filename(
                "fer", "data/haarcascade_frontalface_default.xml"
            )

        if mtcnn:
            try:
                from mtcnn.mtcnn import MTCNN
            except ImportError:
                raise Exception(
                    "MTCNN not installed, install it with pip install mtcnn"
                )
            self.__face_detector = "mtcnn"
            self._mtcnn = MTCNN()
        else:
            self.__face_detector = cv2.CascadeClassifier(cascade_file)

        # Local Keras model
        emotion_model = pkg_resources.resource_filename(
            "fer", "data/emotion_model.hdf5"
        )
        self.__emotion_classifier = load_model(emotion_model, compile=compile)
        self.__emotion_classifier.make_predict_function()
        self.__emotion_target_size = self.__emotion_classifier.input_shape[1:3]

        log.debug("Emotion model: {}".format(emotion_model))

    @staticmethod
    def pad(image):
        row, col = image.shape[:2]
        bottom = image[row - 2 : row, 0:col]
        mean = cv2.mean(bottom)[0]

        bordersize = 40
        padded_image = cv2.copyMakeBorder(
            image,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[mean, mean, mean],
        )
        return padded_image

    @staticmethod
    def depad(image):
        row, col = image.shape[:2]
        return image[40 : row - 40, 40 : col - 40]

    @staticmethod
    def tosquare(bbox):
        """Convert bounding box to square by elongating shorter side."""
        x, y, w, h = bbox
        if h > w:
            diff = h - w
            x -= diff // 2
            w += diff
        elif w > h:
            diff = w - h
            y -= diff // 2
            h += diff
        if w != h:
            log.debug(f"{w} is not {h}")

        return (x, y, w, h)

    def find_faces(self, img: np.ndarray, bgr=True) -> list:
        """Image to list of faces bounding boxes(x,y,w,h)"""
        if isinstance(self.__face_detector, cv2.CascadeClassifier):
            if bgr:
                gray_image_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:  # assume gray
                gray_image_array = img

            faces = self.__face_detector.detectMultiScale(
                gray_image_array,
                scaleFactor=self.__scale_factor,
                minNeighbors=self.__min_neighbors,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=(self.__min_face_size, self.__min_face_size),
            )
        elif self.__face_detector == "mtcnn":
            results = self._mtcnn.detect_faces(img)
            faces = [x["box"] for x in results]

        return faces

    @staticmethod
    def __preprocess_input(x, v2=False):
        x = x.astype("float32")
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
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "sad",
            5: "surprise",
            6: "neutral",
        }

    def detect_emotions(self, img: np.ndarray, face_rectangles: NumpyRects = None) -> list:
        """
        Detects bounding boxes from the specified image with ranking of emotions.
        :param img: image to process (BGR or gray)
        :return: list containing all the bounding boxes detected with their emotions.
        """
        if img is None or not hasattr(img, "shape"):
            raise InvalidImage("Image not valid.")

        emotion_labels = self._get_labels()

        if (face_rectangles is None):
            face_rectangles = self.find_faces(img, bgr=True)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        emotions = []
        for face_coordinates in face_rectangles:
            face_coordinates = self.tosquare(face_coordinates)
            x1, x2, y1, y2 = self.__apply_offsets(face_coordinates)

            if y1 < 0 or x1 < 0:
                gray_img = self.pad(gray_img)
                x1 += 40
                x2 += 40
                y1 += 40
                y2 += 40
                x1 = np.clip(x1, a_min=0, a_max=None)
                y1 = np.clip(y1, a_min=0, a_max=None)

            gray_face = gray_img[y1:y2, x1:x2]

            try:
                gray_face = cv2.resize(gray_face, self.__emotion_target_size)
            except Exception as e:
                log.info("{} resize failed: {}".format(gray_face.shape, e))
                continue
            
            # Local Keras model
            gray_face = self.__preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            emotion_prediction = self.__emotion_classifier.predict(gray_face)[0]
            labelled_emotions = {
                emotion_labels[idx]: round(float(score), 2)
                for idx, score in enumerate(emotion_prediction)
            }

            emotions.append(
                dict(box=face_coordinates, emotions=labelled_emotions)
            )

        self.emotions = emotions

        return emotions

    def top_emotion(self, img: np.ndarray)-> Tuple[Union[str,None], Union[float,None]]:
        """Convenience wrapper for `detect_emotions` returning only top emotion for first face in frame.
        :param img: image to process
        :return: top emotion and score (for first face in frame) or (None, None)

        """
        emotions = self.detect_emotions(img=img)
        top_emotions = [
            max(e["emotions"], key=lambda key: e["emotions"][key]) for e in emotions
        ]

        # Take first face
        if len(top_emotions):
            top_emotion = top_emotions[0]
        else:
            return (None, None)
        score = emotions[0]["emotions"][top_emotion]

        return top_emotion, score


def parse_arguments(args):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Image filepath")
    return parser.parse_args()


def top_emotion():
    args = parse_arguments(sys.argv)
    fer = FER()
    top_emotion, score = fer.top_emotion(args.image)
    print(top_emotion, score)


def main():
    top_emotion()


if __name__ == "__main__":
    main()
