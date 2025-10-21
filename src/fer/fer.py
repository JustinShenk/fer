#!/usr/bin/python3

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
import logging
import sys
from typing import Sequence, Tuple, Union

import cv2
import numpy as np
import pkg_resources
import requests

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    import tensorflow as tf
    from keras.models import load_model


from .utils import load_image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fer")

NumpyRects = Union[np.ndarray, Sequence[Tuple[int, int, int, int]]]

__author__ = "Justin Shenk"

PADDING = 40
SERVER_URL = "http://localhost:8501/v1/models/emotion_model:predict"


class FER:
    """
    Allows performing Facial Expression Recognition ->
        a) Detection of faces
        b) Detection of emotions
    """

    # Class-level model cache for performance
    _model_cache = None
    _model_cache_lock = None
    _tflite_interpreter_cache = None

    def __init__(
        self,
        cascade_file: str = None,
        mtcnn=False,
        tfserving: bool = False,
        use_tflite: bool = True,
        scale_factor: float = 1.1,
        min_face_size: int = 50,
        min_neighbors: int = 5,
        offsets: tuple = (10, 10),
    ):
        """
        Initializes the face detector and Keras model for facial expression recognition.

        Args:
            cascade_file: File URI with the Haar cascade for face classification
            mtcnn: Use MTCNN network for face detection instead of Haar Cascade
            tfserving: Use TensorFlow Serving for predictions
            use_tflite: Use quantized TensorFlow Lite model for 7x faster inference (default: True)
            scale_factor: How much the image size is reduced at each image scale (default: 1.1)
            min_face_size: Minimum size of the face to detect in pixels (default: 50)
            min_neighbors: How many neighbors each candidate rectangle should have (default: 5)
            offsets: Padding around face before classification as (x_offset, y_offset) (default: (10, 10))

        Raises:
            ValueError: If parameters are invalid
            Exception: If MTCNN is requested but facenet-pytorch is not installed
        """
        # Validate parameters
        if scale_factor <= 1.0:
            raise ValueError("scale_factor must be greater than 1.0")
        if min_face_size < 1:
            raise ValueError("min_face_size must be at least 1")
        if min_neighbors < 0:
            raise ValueError("min_neighbors must be non-negative")
        if not isinstance(offsets, (tuple, list)) or len(offsets) != 2:
            raise ValueError("offsets must be a tuple or list of length 2")

        self.__scale_factor = scale_factor
        self.__min_neighbors = min_neighbors
        self.__min_face_size = min_face_size
        self.__offsets = offsets
        self.tfserving = tfserving
        self.use_tflite = use_tflite

        if cascade_file is None:
            cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        if mtcnn:
            try:
                from facenet_pytorch import MTCNN
            except ImportError:
                raise Exception(
                    "MTCNN not installed, install it with pip install facenet-pytorch and from facenet_pytorch import MTCNN"
                )
            self.__face_detector = "mtcnn"

            # use cuda GPU if available
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                device = torch.device('cuda')
                self._mtcnn = MTCNN(keep_all=True, device=device)
            else:
                self._mtcnn = MTCNN(keep_all=True)
        else:
            self.__face_detector = cv2.CascadeClassifier(cascade_file)

        self._initialize_model()

    def _initialize_model(self):
        if self.tfserving:
            self.__emotion_target_size = (64, 64)  # hardcoded for now
        elif self.use_tflite:
            # Use TensorFlow Lite model for faster inference
            self.__emotion_target_size = (64, 64)

            # Use cached interpreter if available
            if FER._tflite_interpreter_cache is not None:
                log.debug("Using cached TFLite interpreter")
                self.__tflite_interpreter = FER._tflite_interpreter_cache
            else:
                tflite_model_path = pkg_resources.resource_filename(
                    "fer", "data/emotion_model_quantized.tflite"
                )
                log.debug(f"Loading TFLite model: {tflite_model_path}")
                self.__tflite_interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
                self.__tflite_interpreter.allocate_tensors()
                # Cache for future instances
                FER._tflite_interpreter_cache = self.__tflite_interpreter

            # Get input and output details
            self.__tflite_input_details = self.__tflite_interpreter.get_input_details()
            self.__tflite_output_details = self.__tflite_interpreter.get_output_details()
        else:
            # Use cached model if available for performance
            if FER._model_cache is not None:
                log.debug("Using cached emotion model")
                self.__emotion_classifier = FER._model_cache
                self.__emotion_target_size = self.__emotion_classifier.input_shape[1:3]
            else:
                # Load model for the first time
                emotion_model = pkg_resources.resource_filename(
                    "fer", "data/emotion_model.hdf5"
                )
                log.debug(f"Loading emotion model: {emotion_model}")
                self.__emotion_classifier = load_model(emotion_model, compile=False)
                self.__emotion_target_size = self.__emotion_classifier.input_shape[1:3]
                # Cache for future instances
                FER._model_cache = self.__emotion_classifier
        return

    def _classify_emotions(self, gray_faces: np.ndarray) -> np.ndarray:  # b x w x h
        """Run faces through online or offline classifier."""
        if self.tfserving:
            gray_faces = np.expand_dims(gray_faces, -1)  # to 4-dimensions
            instances = gray_faces.tolist()
            response = requests.post(SERVER_URL, json={"instances": instances})
            response.raise_for_status()

            emotion_predictions = response.json()["predictions"]
            return emotion_predictions
        elif self.use_tflite:
            # Use TFLite interpreter
            gray_faces = np.expand_dims(gray_faces, -1)  # Add channel dimension
            batch_size = gray_faces.shape[0]

            # TFLite inference
            all_predictions = []
            for i in range(batch_size):
                # Get single face
                face = gray_faces[i:i+1].astype(np.float32)

                # Run inference
                self.__tflite_interpreter.set_tensor(
                    self.__tflite_input_details[0]['index'],
                    face
                )
                self.__tflite_interpreter.invoke()

                # Get output
                output = self.__tflite_interpreter.get_tensor(
                    self.__tflite_output_details[0]['index']
                )
                all_predictions.append(output[0])

            return np.array(all_predictions)
        else:
            return self.__emotion_classifier(gray_faces)

    @staticmethod
    def pad(image):
        """Pad image."""
        row, col = image.shape[:2]
        bottom = image[row - 2 : row, 0:col]
        mean = cv2.mean(bottom)[0]

        padded_image = cv2.copyMakeBorder(
            image,
            top=PADDING,
            bottom=PADDING,
            left=PADDING,
            right=PADDING,
            borderType=cv2.BORDER_CONSTANT,
            value=[mean, mean, mean],
        )
        return padded_image

    @staticmethod
    def depad(image):
        row, col = image.shape[:2]
        return image[PADDING : row - PADDING, PADDING : col - PADDING]

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

    def find_faces(self, img: np.ndarray, bgr=True, gray_img=None) -> list:
        """Image to list of faces bounding boxes(x,y,w,h)"""
        if isinstance(self.__face_detector, cv2.CascadeClassifier):
            # Use provided grayscale image if available to avoid redundant conversion
            if gray_img is not None:
                gray_image_array = gray_img
            elif bgr:
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
            boxes, probs = self._mtcnn.detect(img)
            faces = []
            if type(boxes) == np.ndarray:
                for face in boxes:
                    faces.append(
                        [
                            int(face[0]),
                            int(face[1]),
                            int(face[2]) - int(face[0]),
                            int(face[3]) - int(face[1]),
                        ]
                    )

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
        """Offset face coordinates with padding before classification.
        x1, x2, y1, y2 = 0, 100, 0, 100 becomes -10, 110, -10, 110
        """
        x, y, width, height = face_coordinates
        x_off, y_off = self.__offsets
        x1 = x - x_off
        x2 = x + width + x_off
        y1 = y - y_off
        y2 = y + height + y_off
        return x1, x2, y1, y2

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

    def detect_emotions(
        self, img: np.ndarray, face_rectangles: NumpyRects = None
    ) -> list:
        """
        Detects bounding boxes from the specified image with ranking of emotions.
        :param img: exact image path, numpy array (BGR or gray) or based64 encoded images
        could be passed.
        :return: list containing all the bounding boxes detected with their emotions.
        """
        img = load_image(img)

        emotion_labels = self._get_labels()

        # Convert to grayscale once and reuse
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if face_rectangles is None:
            # Pass grayscale image to find_faces to avoid redundant conversion
            face_rectangles = self.find_faces(img, bgr=True, gray_img=gray_img)

        gray_img = self.pad(gray_img)

        emotions = []
        gray_faces = []
        if face_rectangles is not None:
            # Pre-allocate array for better performance
            gray_faces = []
            valid_face_indices = []

            for idx, face_coordinates in enumerate(face_rectangles):
                face_coordinates = self.tosquare(face_coordinates)

                # offset to expand bounding box
                # Note: x1 and y1 can be negative
                x1, x2, y1, y2 = self.__apply_offsets(face_coordinates)

                # account for padding in bounding box coordinates
                x1 += PADDING
                y1 += PADDING
                x2 += PADDING
                y2 += PADDING
                x1 = max(0, x1)
                y1 = max(0, y1)

                gray_face = gray_img[y1:y2, x1:x2]

                try:
                    gray_face = cv2.resize(gray_face, self.__emotion_target_size)
                except Exception as e:
                    log.warn(f"{gray_face.shape} resize failed: {e}")
                    continue

                gray_faces.append(gray_face)
                valid_face_indices.append(idx)

            # Vectorize preprocessing - process all faces at once
            if gray_faces:
                gray_faces = np.array(gray_faces, dtype="float32")
                # Vectorized preprocessing
                gray_faces = gray_faces / 255.0
                gray_faces = (gray_faces - 0.5) * 2.0

                # Update face_rectangles to only include valid faces
                face_rectangles = [face_rectangles[i] for i in valid_face_indices]

        # predict all faces
        if not len(gray_faces):
            return emotions  # no valid faces

        # classify emotions (gray_faces is already a numpy array)
        emotion_predictions = self._classify_emotions(gray_faces)

        # label scores
        for face_idx, face in enumerate(emotion_predictions):
            labelled_emotions = {
                emotion_labels[idx]: round(float(score), 2)
                for idx, score in enumerate(face)
            }

            emotions.append(
                dict(box=face_rectangles[face_idx], emotions=labelled_emotions)
            )

        self.emotions = emotions

        return emotions

    def batch_detect_emotions(self, imgs: list) -> list:
        """
        Detects emotions from multiple images in a batch for better performance.

        :param imgs: list of images (numpy arrays, file paths, or base64)
        :return: list of results, one per image
        """
        if not imgs:
            return []

        # Process each image individually but could be optimized further
        # by batching face detection and emotion classification across all images
        results = []
        for img in imgs:
            result = self.detect_emotions(img)
            results.append(result)

        return results

    def top_emotion(
        self, img: np.ndarray
    ) -> Tuple[Union[str, None], Union[float, None]]:
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
