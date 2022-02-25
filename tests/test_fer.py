import unittest

import cv2
import pandas as pd

from fer import FER, Video
from fer.exceptions import InvalidImage

detector = None


class TestFER(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global detector, mtcnn_detector
        detector = FER()
        mtcnn_detector = FER(mtcnn=True)

    def test_detect_emotions(self):
        """
        FER is able to detect faces in an image
        :return:
        """
        justin = "justin.jpg"

        result = detector.detect_emotions(justin)  # type: list
        mtcnn_result = mtcnn_detector.detect_emotions(justin)  # type: list

        self.assertEqual(len(result), 1)

        first = result[0]
        mtcnn_first = mtcnn_result[0]

        self.assertGreater(first["emotions"]["happy"], 0.9)
        self.assertGreater(mtcnn_first["emotions"]["happy"], 0.9)
        self.assertIn("box", first)
        self.assertIn("emotions", first)
        self.assertTrue(len(first["box"]), 1)

    def test_detect_faces_invalid_content(self):
        """
        FER detects invalid images
        :return:
        """
        justin = cv2.imread("example.py")

        with self.assertRaises(InvalidImage):
            _ = detector.detect_emotions(justin)  # type: list

    def test_detect_no_faces_on_no_faces_content(self):
        """
        FER successfully reports an empty list when no faces are detected.
        :return:
        """
        justin = cv2.imread("no-faces.jpg")

        result = detector.detect_emotions(justin)  # type: list
        self.assertEqual(len(result), 0)

    def test_top_emotion(self):
        """
        FER successfully returns tuple of string and float for first face.
        :return:
        """
        justin = cv2.imread("justin.jpg")

        top_emotion, score = detector.top_emotion(justin)  # type: tuple
        self.assertIsInstance(top_emotion, str)
        self.assertIsInstance(float(score), float)

    def test_video(self):
        detector = FER()
        video = Video("tests/woman2.mp4")

        raw_data = video.analyze(detector, display=False)
        assert isinstance(raw_data, list)

        # Convert to pandas for analysis
        df = video.to_pandas(raw_data)
        assert (
            sum(df.neutral[:5] > 0.5) == 5
        ), f"Expected neutral > 0.5, got {df.neutral[:5]}"
        assert isinstance(df, pd.DataFrame)
        assert "angry" in df
        df = video.get_first_face(df)
        assert isinstance(df, pd.DataFrame)
        df = video.get_emotions(df)
        assert isinstance(df, pd.DataFrame)

    def tearDownClass():
        global detector
        del detector


if __name__ == "__main__":
    unittest.main()
