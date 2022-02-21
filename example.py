#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import cv2

from fer import FER, draw_annotations

detector = FER(mtcnn=True)  # or with mtcnn=False for Haar Cascade Classifier

try:
    image_path = Path(sys.argv[1])
except IndexError:
    image_path = Path("justin.jpg")

image = cv2.imread(str(image_path.resolve()))
faces = detector.detect_emotions(image)
image = draw_annotations(image, faces)

cv2.imwrite(f"{image_path.stem}_drawn{image_path.suffix}", image)
print(faces)
