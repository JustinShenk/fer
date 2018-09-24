import unittest
import cv2

from fer.exceptions import InvalidImage
from fer.fer import FER

fer = None


class TestFER(unittest.TestCase):

    def setUpClass(self):
        global detector
        detector = FER()

    def test_detect_faces(self):
        """
        FER is able to detect faces and landmarks on an image
        :return:
        """
        justin = cv2.imread("justin.jpg")

        result = detector.detect_emotions(justin)  # type: list

        self.assertEqual(len(result), 1)

        first = result[0]

        self.assertIn('box', first)
        self.assertIn('emotions', first)
        self.assertTrue(len(first['box']), 1)
        self.assertTrue(len(first['keypoints']), 5)

    def test_detect_faces_invalid_content(self):
        """
        FER detects invalid images
        :return:
        """
        justin = cv2.imread("example.py")

        with self.assertRaises(InvalidImage):
            result = detector.detect_faces(justin)  # type: list

    def test_detect_no_faces_on_no_faces_content(self):
        """
        FER successfully reports an empty list when no faces are detected.
        :return:
        """
        ivan = cv2.imread("no-faces.jpg")

        result = detector.detect_faces(ivan)  # type: list
        self.assertEqual(len(result), 0)


    def tearDownClass(self):
        global fer
        del fer

if __name__ == '__main__':
    unittest.main()
