#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

import csv
from typing import Union, List

import cv2
import numpy as np
import os
import pandas as pd
import re
import requests
import tempfile
import time

logging.getLogger(__name__)


def tocap(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return Video(result)

    return wrapper


class Peltarion_Emotion_Classifier(object):
    def __init__(self, url, token, shape=(48, 48)):
        self.url = url
        self.token = token
        self.shape = shape

    @staticmethod
    def unnormalize_face(gray_face: np.ndarray,
                         shape: tuple = (48, 48),
                         v2: bool = True) -> object:
        gray_face = gray_face.reshape(shape)
        if v2:
            gray_face = gray_face / 2.0
            gray_face = gray_face + 0.5
        gray_face = gray_face * 255.0
        return gray_face

    def predict(self, gray_face) -> list:
        """Gray face to emotions with Peltarion REST API"""
        instance_path = os.environ.get('FLASK_INSTANCE_PATH', os.getcwd())
        temp_filepath = os.path.join(instance_path, 'tmp_01.npy')
        gray_face = gray_face.reshape(48, 48, 1).astype(np.float32)
        np.save(temp_filepath, gray_face)
        headers = {'Authorization': 'Bearer ' + self.token}
        files = {'image': open(temp_filepath, 'rb')}
        response = requests.post(self.url, headers=headers, files=files).json()
        try:
            emotion = response['emotion']
        except:
            logging.error(f"{response.text} is not a valid response")
        return emotion


class Video(object):
    def __init__(self, video_file, outdir='output', first_face_only=True, tempfile=None):
        """Video class for extracting and saving frames for emotion detection.
        :param video_file - str
        :param outdir - str
        :param tempdir - str
        :param first_face_only - bool
        :param tempfile - str
        """
        assert os.path.exists(video_file), "Video file not found at {}".format(
            os.path.abspath(video_file))
        self.cap = cv2.VideoCapture(video_file)
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir

        if not first_face_only:
            logging.error("Only single-face charting is implemented")
        self.first_face_only = first_face_only
        self.tempfile = tempfile
        self.filepath = video_file
        self.filename = ''.join(self.filepath.split('/')[-1])

    @staticmethod
    def get_max_faces(data:list):
        max = 0
        for frame in data:
            for face in frame:
                if len(face) > max:
                    max = len(face)
        return max

    def to_dict(self, data:Union[dict, list]) -> dict:
        emotions = []

        frame = data[0]
        if isinstance(frame, list):
            try:
                emotions = frame[0]['emotions'].keys()
            except IndexError:
                raise Exception("No data in 'data'")
        elif isinstance(frame, dict):
            return data

        dictlist = []


        for data_idx, frame in enumerate(data):
            rowdict = {}
            for idx, face in enumerate(list(frame)):
                if not isinstance(face, dict):
                    break
                rowdict.update({'box' + str(idx): face['box']})
                rowdict.update({
                    emo + str(idx): face['emotions'][emo]
                    for emo in emotions
                })
            dictlist.append(rowdict)
        return dictlist

    def to_pandas(self, data:Union[pd.DataFrame, list]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data

        datalist = self.to_dict(data)
        df = pd.DataFrame(datalist)
        if self.first_face_only:
            df = self.get_first_face(df)
        return df

    @staticmethod
    def get_first_face(df:pd.DataFrame) -> pd.DataFrame:
        assert isinstance(df, pd.DataFrame), "Must be a pandas DataFrame"
        try:
            int(df.columns[0][-1])
        except ValueError:
            # Already only one face in df
            return df

        columns = [x for x in df.columns if x[-1] is '0']
        new_columns = [x[:-1] for x in columns]
        single_df = df[columns]
        single_df.columns = new_columns
        return single_df

    @staticmethod
    def get_emotions(df:pd.DataFrame) -> list:
        columns = [x for x in df.columns if 'box' not in x]
        return df[columns]

    def to_csv(self, data, filename='data.csv'):
        def key(item):
            key_pat = re.compile(r"^(\D+)(\d+)$")
            m = key_pat.match(item)
            return m.group(1), int(m.group(2))

        dictlist = self.to_dict(data)
        columns = set().union(*(d.keys() for d in dictlist))
        columns = sorted(columns, key=key) # sort by trailing number (faces)

        with open('data.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, columns, lineterminator='\n')
            writer.writeheader()
            writer.writerows(dictlist)
        return dictlist

    def analyze(
            self,
            detector,
            display:bool=False,
            output:str="csv",
            frequency:int=1,
            max_results:int=None,
            video_id=None,
            save_frames:bool=True,
            save_video:bool=True,
            annotate_frames:bool=True
    ):
        """Recognize facial expressions in video using `detector`."""
        data = []
        frequency = int(frequency)
        results_nr = 0

        # Open video
        assert self.cap.open(self.filepath), "Video capture not opening"
        self.__emotions = detector._get_labels().items()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pos_frames = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        assert int(pos_frames) == 0, "Video not at index 0"

        frameCount = 0
        height, width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert fps and length, "File {} not loaded".format(self.filepath)
        logging.info("{:.2f} fps, {} frames, {:.2f} seconds".format(
            fps, length, length / fps))

        capture_duration = 1000 / fps
        if save_frames:
            os.makedirs(self.outdir, exist_ok=True)

        root, ext = os.path.splitext(self.filepath)
        outfile = os.path.join(self.outdir, f'{root}_output{ext}')

        if save_video:
            videowriter = self.save_video(outfile, fps, width, height)

        while self.cap.isOpened():
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:  # end of video
                break
            if frameCount % frequency != 0:
                frameCount += 1
                continue
            padded_frame = detector.pad(frame)
            try:
                result = detector.detect_emotions(padded_frame)
            except Exception as e:
                logging.error(e)
                break

            # Save images to `self.outdir`
            imgpath = os.path.join(
                self.outdir, (video_id or root) + str(frameCount) + '.jpg')
            if save_frames and not annotate_frames:
                cv2.imwrite(imgpath, frame)

            if display or save_video or annotate_frames:
                assert isinstance(result, list), type(result)
                for face in result:
                    bounding_box = face['box']
                    emotions = face['emotions']

                    cv2.rectangle(frame,
                                  (bounding_box[0] - 40, bounding_box[1] - 40),
                                  (bounding_box[0] - 40 + bounding_box[2],
                                   bounding_box[1] - 40 + bounding_box[3]),
                                  (0, 155, 255), 2)

                    for idx, (emotion, score) in enumerate(emotions.items()):
                        color = (211, 211, 211) if score < 0.01 else (0, 255,
                                                                      0)
                        emotion_score = "{}: {}".format(
                            emotion, "{:.2f}".format(score)
                            if score > 0.01 else "")
                        cv2.putText(frame, emotion_score,
                                    (bounding_box[0] - 40, bounding_box[1] - 40
                                     + bounding_box[3] + 30 + idx * 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                                    cv2.LINE_AA)
                    if display:
                        cv2.imshow('Video', frame)
                    if save_frames and annotate_frames:
                        cv2.imwrite(imgpath, frame)
                    if save_video:
                        videowriter.write(frame)
                    results_nr += 1

                if display or save_video:
                    remaining_duration = max(
                        1,
                        int((time.time() - start_time) * 1000 -
                            capture_duration))
                    if cv2.waitKey(remaining_duration) & 0xFF == ord('q'):
                        break
                else:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            frameCount += 1
            if result:
                data.append(result)
            if max_results and results_nr > max_results:
                break

        self.cap.release()
        if display or save_video:
            videowriter.release()
            logging.info("Completed analysis: saved to {}".format(self.tempfile
                                                                  or outfile))
            if self.tempfile:
                os.replace(self.tempfile, outfile)

        if output == 'csv':
            return self.to_csv(data)
        elif output == 'pandas':
            return self.to_pandas(data)
        else:
            raise NotImplementedError(f"{output} is not supported")
        return data

    def save_video(self, outfile, fps:int, width:int, height:int):
        if os.path.isfile(outfile):
            os.remove(outfile)
            logging.info("Deleted pre-existing {}".format(outfile))
        if self.tempfile and os.path.isfile(self.tempfile):
            os.remove(self.tempfile)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videowriter = cv2.VideoWriter(self.tempfile or outfile, fourcc, fps,
                                      (width, height), True)
        return videowriter

    def __del__(self):
        try:
            cv2.destroyAllWindows()
        except:
            pass
