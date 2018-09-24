#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from fer.fer import FER

detector = FER()

image = cv2.imread("justin.jpg")
result = detector.detect_emotions(image)

# Result is an array with all the bounding boxes detected. We know that for 'justin.jpg' there is only one.
bounding_box = result[0]['box']
emotions = result[0]['emotions']

cv2.rectangle(image,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255),
              2)

for idx, (emotion, score) in enumerate(emotions.items()):
    if score < 0.01:
        continue
    emotion_score = "{}: {:.2f}".format(emotion, score)
    cv2.putText(image,
                emotion_score,
                (bounding_box[0],bounding_box[1]+bounding_box[3] + 30 + idx*15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                1,
                cv2.LINE_AA)
cv2.imwrite("justin_drawn.jpg", image)

print(result)