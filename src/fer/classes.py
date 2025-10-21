#!/usr/bin/env python3
import csv
import logging
import os
import re
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional, Union
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .utils import draw_annotations

log = logging.getLogger("fer")

# Optional moviepy import - only needed for audio features
try:
    from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip
    MOVIEPY_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    MOVIEPY_AVAILABLE = False
    log.warning(f"moviepy not available: {e}. Audio features will be disabled.")


class AsyncFrameWriter:
    """Asynchronous frame writer for non-blocking I/O operations."""

    def __init__(self, max_queue_size=50):
        """Initialize async frame writer with a background thread.

        Args:
            max_queue_size: Maximum number of frames to buffer in queue
        """
        self.queue = Queue(maxsize=max_queue_size)
        self.thread = Thread(target=self._write_worker, daemon=True)
        self.running = False

    def start(self):
        """Start the background writer thread."""
        self.running = True
        self.thread.start()

    def _write_worker(self):
        """Background worker that writes frames from queue."""
        while self.running or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.1)
                if item is None:  # Poison pill
                    break

                filepath, frame = item
                cv2.imwrite(filepath, frame)
                self.queue.task_done()
            except:
                continue

    def write(self, filepath: str, frame: np.ndarray):
        """Queue a frame for async writing.

        Args:
            filepath: Path where frame should be saved
            frame: Frame data to save
        """
        self.queue.put((filepath, frame))

    def stop(self):
        """Stop the writer and wait for queue to empty."""
        self.queue.join()  # Wait for all items to be processed
        self.running = False
        self.queue.put(None)  # Poison pill
        self.thread.join(timeout=5.0)


class Video:
    def __init__(
        self,
        video_file: str,
        outdir: str = "output",
        first_face_only: bool = True,
        tempfile: Optional[str] = None,
    ):
        """Video class for extracting and saving frames for emotion detection.
        param video_file - str
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
        max_faces = 0
        for frame in data:
            for face in frame:
                if len(face) > max_faces:
                    max_faces = len(face)
        return max_faces

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
        """Get emotion columns from results."""
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
            log.info(f"Completed analysis: saved to {self.tempfile or outfile}")
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
        self, frame, faces, video_id, root, lang="en", size_multiplier=1, async_writer=None
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
            if async_writer is not None:
                # Use async I/O for non-blocking write
                async_writer.write(imgpath, frame.copy())
            else:
                # Fallback to synchronous write
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
        batch_size: int = 1,
        use_async_io: bool = True,
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
            batch_size (int): number of frames to process together for GPU efficiency (default: 1)
            use_async_io (bool): use asynchronous I/O for frame saving (default: True)
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
        assert fps and length, f"File {self.filepath} not loaded"

        if save_fps is not None:
            frequency = fps // save_fps
            log.info(f"Saving every {frequency} frames")

        log.info(
            f"{fps:.2f} fps, {length} frames, {length / fps:.2f} seconds"
        )

        if self.save_frames:
            os.makedirs(self.outdir, exist_ok=True)
            log.info(f"Making directories at {self.outdir}")
        root, ext = os.path.splitext(os.path.basename(self.filepath))
        outfile = os.path.join(self.outdir, f"{root}_output{ext}")

        if save_video:
            self.videowriter = self._save_video(outfile, fps, width, height)

        total_frames = length
        if frequency > 1:
            total_frames = length // frequency

        # Initialize async frame writer if enabled
        async_writer = None
        if use_async_io and save_frames:
            async_writer = AsyncFrameWriter(max_queue_size=batch_size * 2)
            async_writer.start()
            log.info("Async I/O enabled for frame saving")

        # Frame batching setup
        frame_batch = []
        frame_metadata = []  # Store (frame_number, detection_box) for each frame in batch

        with logging_redirect_tqdm():
            pbar = tqdm(total=total_frames, unit="frames")

        try:
            while self.cap.isOpened():
                # Optimize frame skipping by seeking directly to target frames
                if frequency > 1:
                    target_frame = self.frameCount
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

                ret, frame = self.cap.read()
                if not ret:  # end of video
                    break

                if frame is None:
                    log.warn("Empty frame")
                    if frequency > 1:
                        self.frameCount += frequency
                    else:
                        self.frameCount += 1
                    continue

                if detection_box is not None:
                    frame = self._crop(frame, detection_box)

                # Add frame to batch
                frame_batch.append(frame.copy())
                frame_metadata.append((self.frameCount, detection_box))

                # Process batch when full or at end of video
                if len(frame_batch) >= batch_size or (max_results and results_nr + len(frame_batch) >= max_results):
                    # Process batch of frames efficiently
                    try:
                        if batch_size > 1:
                            # Use batch processing for multiple frames
                            batch_results = detector.batch_detect_emotions(frame_batch)
                        else:
                            # Single frame - use regular detection
                            batch_results = [detector.detect_emotions(frame_batch[0])]
                    except Exception as e:
                        log.error(e)
                        break

                    # Process results for each frame in batch
                    for idx, (batch_frame, (frame_num, det_box), faces) in enumerate(zip(frame_batch, frame_metadata, batch_results)):
                        # Offset detection_box to include padding
                        if det_box is not None:
                            faces = self._offset_detection_box(faces, det_box)

                        self._increment_frames(batch_frame, faces, video_id, root, lang, size_multiplier, async_writer)

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                        if faces:
                            frames_emotions.append(faces)

                        results_nr += 1
                        pbar.update(1)

                        # Advance frameCount by frequency for next iteration
                        if frequency > 1:
                            self.frameCount += frequency - 1  # -1 because _increment_frames already added 1

                        if max_results and results_nr >= max_results:
                            break

                    # Clear batch
                    frame_batch = []
                    frame_metadata = []

                    if max_results and results_nr >= max_results:
                        break

            # Process remaining frames in batch
            if frame_batch:
                try:
                    if len(frame_batch) > 1:
                        batch_results = detector.batch_detect_emotions(frame_batch)
                    else:
                        batch_results = [detector.detect_emotions(frame_batch[0])]
                except Exception as e:
                    log.error(e)
                else:
                    for idx, (batch_frame, (frame_num, det_box), faces) in enumerate(zip(frame_batch, frame_metadata, batch_results)):
                        if det_box is not None:
                            faces = self._offset_detection_box(faces, det_box)

                        self._increment_frames(batch_frame, faces, video_id, root, lang, size_multiplier, async_writer)

                        if faces:
                            frames_emotions.append(faces)

                        results_nr += 1
                        pbar.update(1)

                        if frequency > 1:
                            self.frameCount += frequency - 1

        finally:
            pbar.close()
            # Stop async writer if enabled
            if async_writer is not None:
                log.info("Waiting for async I/O to complete...")
                async_writer.stop()
                log.info("Async I/O completed")

        self._close_video(outfile, save_frames, zip_images)

        if include_audio:
            if not MOVIEPY_AVAILABLE:
                log.error("Audio feature requested but moviepy is not available. Install with: pip install moviepy")
            else:
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
            log.info(f"Deleted pre-existing {outfile}")
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
