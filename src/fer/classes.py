#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import logging
import os
import re
from moviepy.editor import *
from pathlib import Path
from typing import Optional, Union
from zipfile import ZipFile

import cv2
import pandas as pd

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .utils import draw_annotations

log = logging.getLogger("fer")


class Video(object):
    def __init__(
        self,
        video_file: str,
        outdir: str = "output",
        first_face_only: bool = True,
        tempfile: Optional[str] = None,
    ):
        """Video class for extracting and saving frames for emotion detection.
        :param video_file - str
        :param outdir - str
        :param tempdir - str
        :param first_face_only - bool
        :param tempfile - str
        """
        assert os.path.exists(video_file), "Video file not found at {}".format(
            os.path.abspath(video_file)
        )
        self.cap = cv2.VideoCapture(video_file)
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir

        if not first_face_only:
            log.error("Only single-face charting is implemented")
        self.first_face_only = first_face_only
        self.tempfile = tempfile
        self.filepath = video_file
        self.filename = "".join(self.filepath.split("/")[-1])

    @staticmethod
    def get_max_faces(data: list) -> int:
        """Get max number of faces detected in a series of frames, eg 3"""
        max = 0
        for frame in data:
            for face in frame:
                if len(face) > max:
                    max = len(face)
        return max

    @staticmethod
    def _to_dict(data: Union[dict, list]) -> dict:
        emotions = []

        frame = data[0]
        if isinstance(frame, list):
            try:
                emotions = frame[0]["emotions"].keys()
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
                rowdict.update({"box" + str(idx): face["box"]})
                rowdict.update(
                    {emo + str(idx): face["emotions"][emo] for emo in emotions}
                )
            dictlist.append(rowdict)
        return dictlist

    def to_pandas(self, data: Union[pd.DataFrame, list]) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        if isinstance(data, pd.DataFrame):
            return data

        if not len(data):
            return pd.DataFrame()
        datalist = self._to_dict(data)
        df = pd.DataFrame(datalist)
        if self.first_face_only:
            df = self.get_first_face(df)
        return df

    @staticmethod
    def get_first_face(df: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(df, pd.DataFrame), "Must be a pandas DataFrame"
        try:
            int(df.columns[0][-1])
        except ValueError:
            # Already only one face in df
            return df

        columns = [x for x in df.columns if x[-1] == "0"]
        new_columns = [x[:-1] for x in columns]
        single_df = df[columns]
        single_df.columns = new_columns
        return single_df

    @staticmethod
    def get_emotions(df: pd.DataFrame) -> list:
        """Get emotion columsn from results."""
        columns = [x for x in df.columns if "box" not in x]
        return df[columns]

    def to_csv(self, data, filename="data.csv"):
        """Save data to csv"""

        def key(item):
            key_pat = re.compile(r"^(\D+)(\d+)$")
            m = key_pat.match(item)
            return m.group(1), int(m.group(2))

        dictlist = self._to_dict(data)
        columns = set().union(*(d.keys() for d in dictlist))
        columns = sorted(columns, key=key)  # sort by trailing number (faces)

        with open("data.csv", "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(dictlist)
        return dictlist

    def _close_video(self, outfile, save_frames, zip_images):
        self.cap.release()
        if self.display or self.save_video:
            self.videowriter.release()

        if self.save_video:
            log.info("Completed analysis: saved to {}".format(self.tempfile or outfile))
            if self.tempfile:
                os.replace(self.tempfile, outfile)

        if save_frames and zip_images:
            log.info("Starting to Zip")
            outdir = Path(self.outdir)
            zip_dir = outdir / "images.zip"
            images = sorted(list(outdir.glob("*.jpg")))
            total = len(images)
            i = 0
            with ZipFile(zip_dir, "w") as zip:
                for file in images:
                    zip.write(file, arcname=file.name)
                    os.remove(file)
                    i += 1
                    if i % 50 == 0:
                        log.info(f"Compressing: {i*100 // total}%")
            log.info("Zip has finished")

    def _offset_detection_box(self, faces, detection_box):
        for face in faces:
            original_box = face.get("box")
            face["box"] = (
                original_box[0] + detection_box.get("x_min"),
                original_box[1] + detection_box.get("y_min"),
                original_box[2],
                original_box[3],
            )
        return faces

    def _increment_frames(
        self, frame, faces, video_id, root, lang="en", size_multiplier=1
    ):
        # Save images to `self.outdir`
        imgpath = os.path.join(
            self.outdir, (video_id or root) + str(self.frameCount) + ".jpg"
        )

        if self.annotate_frames:
            frame = draw_annotations(
                frame,
                faces,
                boxes=True,
                scores=True,
                lang=lang,
                size_multiplier=size_multiplier,
            )

        if self.save_frames:
            cv2.imwrite(imgpath, frame)

        if self.display:
            cv2.imshow("Video", frame)

        if self.save_video:
            self.videowriter.write(frame)

        self.frameCount += 1

    def analyze(
        self,
        detector,  # fer.FER instance
        display: bool = False,
        output: str = "csv",
        frequency: Optional[int] = None,
        max_results: int = None,
        save_fps: Optional[int] = None,
        video_id: Optional[str] = None,
        save_frames: bool = True,
        save_video: bool = True,
        annotate_frames: bool = True,
        zip_images: bool = True,
        detection_box: Optional[dict] = None,
        lang: str = "en",
        include_audio: bool = False,
        size_multiplier: int = 1,
    ) -> list:
        """Recognize facial expressions in video using `detector`.

        Args:

            detector (fer.FER): facial expression recognizer
            display (bool): show images with cv2.imshow
            output (str): csv or pandas
            frequency (int): inference on every nth frame (higher number is faster)
            max_results (int): number of frames to run inference before stopping
            save_fps (bool): inference frequency = video fps // save_fps
            video_id (str): filename for saving
            save_frames (bool): saves frames to directory
            save_video (bool): saves output video
            annotate_frames (bool): add emotion labels
            zip_images (bool): compress output
            detection_box (dict): dict with bounding box for subimage (xmin, xmax, ymin, ymax)
            lang (str): emotion language that will be shown on video
            include_audio (bool): indicates if a sounded version of the prediction video should be created or not
            size_multiplier (int): increases the size of emotion labels shown in the video by x(size_multiplier)
        Returns:

            data (list): list of results

        """
        frames_emotions = []
        if frequency is None:
            frequency = 1
        else:
            frequency = int(frequency)

        self.display = display
        self.save_frames = save_frames
        self.save_video = save_video
        self.annotate_frames = annotate_frames

        results_nr = 0

        # Open video
        assert self.cap.open(self.filepath), "Video capture not opening"
        self.__emotions = detector._get_labels().items()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pos_frames = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        assert int(pos_frames) == 0, "Video not at index 0"

        self.frameCount = 0
        height, width = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert fps and length, "File {} not loaded".format(self.filepath)

        if save_fps is not None:
            frequency = fps // save_fps
            log.info("Saving every {} frames".format(frequency))

        log.info(
            "{:.2f} fps, {} frames, {:.2f} seconds".format(fps, length, length / fps)
        )

        if self.save_frames:
            os.makedirs(self.outdir, exist_ok=True)
            log.info(f"Making directories at {self.outdir}")
        root, ext = os.path.splitext(os.path.basename(self.filepath))
        outfile = os.path.join(self.outdir, f"{root}_output{ext}")

        if save_video:
            self.videowriter = self._save_video(outfile, fps, width, height)

        with logging_redirect_tqdm():
            pbar = tqdm(total=length, unit="frames")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:  # end of video
                break

            if frame is None:
                log.warn("Empty frame")
                continue

            if self.frameCount % frequency != 0:
                self.frameCount += 1
                continue

            if detection_box is not None:
                frame = self._crop(frame, detection_box)

            # Get faces and detect emotions; coordinates are for unpadded frame
            try:
                faces = detector.detect_emotions(frame)
            except Exception as e:
                log.error(e)
                break

            # Offset detection_box to include padding
            if detection_box is not None:
                faces = self._offset_detection_box(faces, detection_box)

            self._increment_frames(frame, faces, video_id, root, lang, size_multiplier)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if faces:
                frames_emotions.append(faces)

            results_nr += 1
            if max_results and results_nr > max_results:
                break

            pbar.update(1)

        pbar.close()
        self._close_video(outfile, save_frames, zip_images)

        if include_audio:
            audio_suffix = "_audio."
            my_audio = AudioFileClip(self.filepath)
            new_audioclip = CompositeAudioClip([my_audio])

            my_output_clip = VideoFileClip(outfile)
            my_output_clip.audio = new_audioclip
            my_output_clip.write_videofile(audio_suffix.join(outfile.rsplit(".", 1)))

        return self.to_format(frames_emotions, output)

    def to_format(self, data, format):
        """Return data in format."""
        methods_lookup = {"csv": self.to_csv, "pandas": self.to_pandas}
        return methods_lookup[format](data)

    def _save_video(self, outfile: str, fps: int, width: int, height: int):
        if os.path.isfile(outfile):
            os.remove(outfile)
            log.info("Deleted pre-existing {}".format(outfile))
        if self.tempfile and os.path.isfile(self.tempfile):
            os.remove(self.tempfile)
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        videowriter = cv2.VideoWriter(
            self.tempfile or outfile, fourcc, fps, (width, height), True
        )
        return videowriter

    @staticmethod
    def _crop(frame, detection_box):
        crop_frame = frame[
            detection_box.get("y_min") : detection_box.get("y_max"),
            detection_box.get("x_min") : detection_box.get("x_max"),
        ]
        return crop_frame

    def __del__(self):
        cv2.destroyAllWindows()
