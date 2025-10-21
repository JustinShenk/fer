import base64
import os

import cv2
import numpy as np
import requests
from PIL import Image

from .emotionsmultilanguage import emotions_dict
from .exceptions import InvalidImage


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
                (x, y),
                (x + w, y + h),
                color,
                2,
            )

        if scores:
            frame = draw_scores(frame, emotions, (x, y, w, h), lang, size_multiplier)
    return frame


def loadBase64Img(uri):
    """Load image from base64-encoded URI.

    Args:
        uri: Base64-encoded image string with data URI prefix

    Returns:
        numpy.ndarray: Decoded image in BGR format
    """
    encoded_data = uri.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def pil_to_bgr(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def load_image(img):
    """Load image from various sources.

    Modified from github.com/serengil/deepface.

    Args:
        img: Can be a numpy array, file path, base64-encoded string, or URL

    Returns:
        numpy.ndarray: Image in BGR format (opencv-style)

    Raises:
        InvalidImage: If image cannot be loaded or is invalid
        ValueError: If file path doesn't exist
    """
    is_exact_image = is_base64_img = is_url_img = False

    if type(img).__module__ == np.__name__:
        is_exact_image = True
    elif img is None:
        raise InvalidImage("Image not valid.")
    elif isinstance(img, str):
        if len(img) > 11 and img[0:11] == "data:image/":
            is_base64_img = True
        elif len(img) > 11 and img.startswith("http"):
            is_url_img = True
    else:
        raise InvalidImage(f"Unsupported image type: {type(img)}")

    try:
        if is_base64_img:
            img = loadBase64Img(img)
        elif is_url_img:
            response = requests.get(img, stream=True, timeout=10)
            response.raise_for_status()
            img = pil_to_bgr(Image.open(response.raw))
        elif not is_exact_image:  # image path passed as input
            if not os.path.isfile(img):
                raise ValueError(f"Confirm that {img} exists")
            img = cv2.imread(img)
            if img is None:
                raise InvalidImage(f"Failed to read image from {img}")
    except requests.RequestException as e:
        raise InvalidImage(f"Failed to download image from URL: {e}")
    except Exception as e:
        raise InvalidImage(f"Failed to load image: {e}")

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
            emotion, f"{score:.2f}" if score >= 0.01 else ""
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
