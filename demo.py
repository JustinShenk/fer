#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import click
import cv2
from matplotlib import pyplot as plt

from fer import FER
from fer.utils import draw_annotations
from fer.classes import Video

mtcnn_help = "Use slower but more accurate mtcnn face detector"


@click.group()
def cli():
    pass


@cli.command()
@click.argument("device", default=0, type=int)
@click.option("--mtcnn", is_flag=True, help=mtcnn_help)
def webcam(device, mtcnn):
    """Detect emotions from webcam feed. DEVICE is the webcam device number (usually 0 or 1)."""
    detector = FER(mtcnn=mtcnn)
    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.flip(frame, 1)
        emotions = detector.detect_emotions(frame)
        frame = draw_annotations(frame, emotions)

        # Display the resulting frame
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


@cli.command()
@click.argument("image_path", default="justin.jpg", type=click.Path(exists=True))
@click.option("--mtcnn", is_flag=True, help=mtcnn_help)
def image(image_path, mtcnn):
    """Detect emotions in an image file. IMAGE_PATH is the path to the image (default: justin.jpg)."""
    image_path = Path(image_path)
    detector = FER(mtcnn=mtcnn)

    image = cv2.imread(str(image_path.resolve()))
    faces = detector.detect_emotions(image)
    image = draw_annotations(image, faces)

    outpath = f"{image_path.stem}_drawn{image_path.suffix}"
    cv2.imwrite(outpath, image)
    print(f"{faces}\nSaved to {outpath}")


@cli.command()
@click.argument("video_file", default="tests/test.mp4", type=click.Path(exists=True))
@click.option("--mtcnn", is_flag=True, help=mtcnn_help)
def video(video_file, mtcnn):
    """Analyze emotions in a video file. VIDEO_FILE is the path to the video (default: tests/test.mp4)."""
    video = Video(video_file)
    detector = FER(mtcnn=mtcnn)

    # Output list of dictionaries
    raw_data = video.analyze(detector, display=False)

    # Convert to pandas for analysis
    df = video.to_pandas(raw_data)
    df = video.get_first_face(df)
    df = video.get_emotions(df)

    # Plot emotions
    df.plot()
    plt.show()


if __name__ == "__main__":
    cli()
