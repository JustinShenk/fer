#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import cv2
import logging
import os
import pandas as pd
import re
import time


def tocap(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return Video(result)

    return wrapper


class Video(object):
    def __init__(self, video_file, outdir='output', tempfile=None):
        """Video class.
        :param video_file - str
        :param outdir - str
        :param tempfile - str
        """
        assert os.path.exists(video_file), "Video file not found at {}".format(os.path.abspath(video_file))
        self.cap = cv2.VideoCapture(video_file)
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir
        self.tempfile = tempfile
        self.__video_file = video_file

    @staticmethod
    def get_max_faces(data):
        max = 0
        for frame in data:
            for face in frame:
                if len(face) > max:
                    max = len(face)
        return max

    def filepath(self):
        return self.__video_file

    def filename(self):
        return ''.join(self.filepath().split('/')[-1])

    def to_dict(self, data):
        max_faces = self.get_max_faces(data)
        emotions = []
        for frame in data:
            try:
                emotions = frame[0]['emotions'].keys()
            except IndexError:
                pass
        unique_columns = ['box' + str(idx) for idx in range(max_faces)] + \
                         [emo + str(idx) for emo in emotions for idx in range(max_faces)]

        # datadict = {column:[] for column in unique_columns}
        dictlist = []
        for data_idx, frame in enumerate(data):
            # for idx, face in enumerate(frame):
            #     datadict['box' + str(idx)].append(face['box'])
            #     [datadict[emo + str(idx)].append(score) for emo, score in face['emotions'].items()]

            rowdict = {}
            for idx, face in enumerate(frame):
                rowdict.update({'box' + str(idx): face['box']})
                rowdict.update({emo + str(idx): face['emotions'][emo] for emo in emotions})
            dictlist.append(rowdict)
        return dictlist

    def to_pandas(self, data):
        datalist = self.to_dict(data)
        df = pd.DataFrame(datalist)
        return df

    @staticmethod
    def get_first_face(df):
        assert isinstance(df, pd.DataFrame), "Must be a pandas DataFrame"
        columns = [x for x in df.columns if x[-1] is '0']
        new_columns = [x[:-1] for x in columns]
        single_df = df[columns]
        single_df.columns = new_columns
        return single_df

    @staticmethod
    def get_emotions(df):
        columns = [x for x in df.columns if 'box' not in x]
        return df[columns]

    def to_csv(self, data, filename='data.csv'):
        key_pat = re.compile(r"^(\D+)(\d+)$")

        def key(item):
            m = key_pat.match(item)
            return m.group(1), int(m.group(2))

        dictlist = self.to_dict(data)
        max_faces = max([len(faces) for faces in data])
        columns = set().union(*(d.keys() for d in dictlist))
        columns.sort(key=key)  # sort by trailing number (faces)

        with open('data.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, columns, lineterminator='\n')
            writer.writeheader()
            writer.writerows(dictlist)
        return True

    def analyze(self, detector, display=False, output=None, frequency=4, save_frames=True, save_video=True, annotate_frames=True):
        data = []

        assert self.cap.open(self.filepath()), "Video capture not opening"
        self.__emotions = detector._get_labels().items()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pos_frames = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        assert int(pos_frames) == 0, "Video not at index 0"

        frameCount = 0
        height, width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert fps and length, "File {} not loaded".format(self.filepath())
        logging.info("{:.2f} fps, {} frames, {:.2f} seconds".format(fps, length, length / fps))

        capture_duration = 1000 / fps
        if save_frames:
            os.makedirs(self.outdir, exist_ok=True)

        if save_video:
            outfile = os.path.join(self.outdir, 'output.mp4')
            if os.path.isfile(outfile):
                os.remove(outfile); logging.info("Deleted pre-existing {}".format(outfile))
            if self.tempfile and os.path.isfile(self.tempfile):
                os.remove(self.tempfile)
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            out = cv2.VideoWriter(self.tempfile or outfile, fourcc, fps, (width, height), True)

        while self.cap.isOpened():
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break
            if frameCount % frequency != 0:
                frameCount += 1
                continue
            padded_frame = detector.pad(frame)
            try:
                result = detector.detect_emotions(padded_frame)
            except Exception as e:
                print("ERROR: {}".format(e))
                break

            if save_frames and not annotate_frames:
                name = os.path.join(self.outdir,'frame' + str(frameCount) + '.jpg')
                cv2.imwrite(name, frame)

            if display or save_video or annotate_frames:
                for face in result:
                    bounding_box = face['box']
                    emotions = face['emotions']

                    cv2.rectangle(frame,
                                  (bounding_box[0]-40, bounding_box[1]-40),
                                  (bounding_box[0]-40 + bounding_box[2], bounding_box[1]-40 + bounding_box[3]),
                                  (0, 155, 255),
                                  2)

                    for idx, (emotion, score) in enumerate(emotions.items()):
                        color = (211, 211, 211) if score < 0.01 else (0, 255, 0)
                        emotion_score = "{}: {}".format(emotion, "{:.2f}".format(score) if score > 0.01 else "")
                        cv2.putText(frame,
                                    emotion_score,
                                    (bounding_box[0]-40, bounding_box[1]-40 + bounding_box[3] + 30 + idx * 15),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    1,
                                    cv2.LINE_AA)
                    if display:
                        cv2.imshow('Video', frame)
                    if save_frames and annotate_frames:
                        name = os.path.join(self.outdir, 'frame' + str(frameCount) + '.jpg')
                        cv2.imwrite(name, frame)

                    if save_video:
                        out.write(frame)

                if display or save_video:
                    remaining_duration = max(1, int((time.time() - start_time) * 1000 - capture_duration))
                    if cv2.waitKey(remaining_duration) & 0xFF == ord('q'):
                        break
                else:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            frameCount += 1
            data.append(result)

        self.cap.release()
        if display or save_video:
            out.release()
            logging.info("Completed analysis: saved to {}".format(self.tempfile or outfile))
            if self.tempfile:
                os.replace(self.tempfile, outfile)

        if output is 'csv':
            return self.to_csv(data)
        elif output is 'pandas':
            return self.to_pandas(data)

        return data

    def __del__(self):
        cv2.destroyAllWindows()