#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import cv2
from fer.fer import FER
from fer.classes import Video
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys


if __name__ == '__main__':
    try:
        videofile = sys.argv[1]
    except:
        videofile = 'test.mp4'
    detector = FER()
    video = Video(videofile)

    raw_data = video.analyze(detector, display=True)

    # Convert to pandas for analysis
    df = video.to_pandas(raw_data)
    df = video.get_first_face(df)
    df = video.get_emotions(df)

    # Plot emotions
    df.plot()
    plt.show()
