import base64
import os
import requests

import cv2
import numpy as np
from PIL import Image

from .exceptions import InvalidImage
from .emotionsmultilanguage import emotions_dict


def draw_annotations(
    frame: np.ndarray,
    faces: list,
    boxes=True,
    scores=True,
    color: tuple = (0, 155, 255),
    lang: str = "en",
    size_multiplier: int = 1,
) -> np.ndarray:
    """Draws boxes around detected faces. Faces is a list of dicts with `box` and `emotions`."""
    if not len(faces):
        return frame

    for face in faces:
        x, y, w, h = face["box"]
        emotions = face["emotions"]

        if boxes:
            cv2.rectangle(
                frame,
                (x, y, w, h),
                color,
                2,
            )

        if scores:
            frame = draw_scores(frame, emotions, (x, y, w, h), lang, size_multiplier)
    return frame


def loadBase64Img(uri):
    encoded_data = uri.split(",")[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def pil_to_bgr(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def load_image(img):
    """Modified from github.com/serengil/deepface. Returns bgr (opencv-style) numpy array."""
    is_exact_image = is_base64_img = is_url_img = False

    if type(img).__module__ == np.__name__:
        is_exact_image = True
    elif img is None:
        raise InvalidImage("Image not valid.")
    elif len(img) > 11 and img[0:11] == "data:image/":
        is_base64_img = True
    elif len(img) > 11 and img.startswith("http"):
        is_url_img = True

    if is_base64_img:
        img = loadBase64Img(img)
    elif is_url_img:
        img = pil_to_bgr(Image.open(requests.get(img, stream=True).raw))
    elif not is_exact_image:  # image path passed as input
        if not os.path.isfile(img):
            raise ValueError(f"Confirm that {img} exists")
        img = cv2.imread(img)

    if img is None or not hasattr(img, "shape"):
        raise InvalidImage("Image not valid.")

    return img


def draw_scores(
    frame: np.ndarray,
    emotions: dict,
    bounding_box: dict,
    lang: str = "en",
    size_multiplier: int = 1,
) -> np.ndarray:
    """Draw scores for each emotion under faces."""
    GRAY = (211, 211, 211)
    GREEN = (0, 255, 0)
    x, y, w, h = bounding_box

    for idx, (emotion, score) in enumerate(emotions.items()):
        color = GRAY if score < 0.01 else GREEN

        if lang != "en":
            emotion = emotions_dict[emotion][lang]

        emotion_score = "{}: {}".format(
            emotion, "{:.2f}".format(score) if score >= 0.01 else ""
        )
        cv2.putText(
            frame,
            emotion_score,
            (
                x,
                y + h + (15 * size_multiplier) + idx * (15 * size_multiplier),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * size_multiplier,
            color,
            1 * size_multiplier,
            cv2.LINE_AA,
        )
    return frame
