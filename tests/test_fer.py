import unittest
import cv2
import pandas as pd

from fer.classes import Video
from fer.exceptions import InvalidImage
from fer.fer import FER

detector = None


class TestFER(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global detector
        detector = FER()

    def test_detect_emotions(self):
        """
        FER is able to detect faces image
        :return:
        """
        justin = cv2.imread("justin.jpg")

        result = detector.detect_emotions(justin)  # type: list

        self.assertEqual(len(result), 1)

        first = result[0]

        self.assertIn('box', first)
        self.assertIn('emotions', first)
        self.assertTrue(len(first['box']), 1)

    def test_detect_faces_invalid_content(self):
        """
        FER detects invalid images
        :return:
        """
        justin = cv2.imread("example.py")

        with self.assertRaises(InvalidImage):
            result = detector.detect_emotions(justin)  # type: list

    def test_detect_no_faces_on_no_faces_content(self):
        """
        FER successfully reports an empty list when no faces are detected.
        :return:
        """
        justin = cv2.imread("no-faces.jpg")

        result = detector.detect_emotions(justin)  # type: list
        self.assertEqual(len(result), 0)

    def test_video(self):
        detector = FER()
        video = Video("tests/woman2.mp4")

        raw_data = video.analyze(detector, display=False)
        assert isinstance(raw_data, list)

        # Convert to pandas for analysis
        df = video.to_pandas(raw_data)
        assert isinstance(df, pd.DataFrame)
        assert 'angry' in df
        df = video.get_first_face(df)
        assert isinstance(df, pd.DataFrame)
        df = video.get_emotions(df)
        assert isinstance(df, pd.DataFrame)

    def tearDownClass():
        global detector
        del detector


if __name__ == '__main__':
    unittest.main()
