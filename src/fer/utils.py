import cv2
import numpy as np


def draw_annotations(
    frame: np.ndarray,
    faces: list,
    boxes=True,
    scores=True,
    color: tuple = (0, 155, 255),
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
            frame = draw_scores(frame, emotions, (x, y, w, h))
    return frame


def draw_scores(frame: np.ndarray, emotions: dict, bounding_box: dict) -> np.ndarray:
    """Draw scores for each emotion under faces."""
    GRAY = (211, 211, 211)
    GREEN = (0, 255, 0)
    x, y, w, h = bounding_box

    for idx, (emotion, score) in enumerate(emotions.items()):
        color = GRAY if score < 0.01 else GREEN
        emotion_score = "{}: {}".format(
            emotion, "{:.2f}".format(score) if score >= 0.01 else ""
        )
        cv2.putText(
            frame,
            emotion_score,
            (
                x,
                y + h + 15 + idx * 15,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame
