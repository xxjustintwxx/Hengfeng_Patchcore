"""
Capture a single snapshot from IP Webcam (Android) via HTTP.

Usage:
    from capture import capture_snapshot
    bgr = capture_snapshot()  # returns H×W×3 BGR numpy array
"""
import time

import cv2
import numpy as np
import requests


def capture_snapshot(
    url: str = "http://localhost:8080/shot.jpg",
    timeout: float = 3.0,
    retries: int = 3,
    retry_delay: float = 0.5,
    pre_delay: float = 0.3,
) -> np.ndarray:
    """Fetch one JPEG frame from IP Webcam and return it as a BGR numpy array.

    Args:
        url:         HTTP endpoint that returns a single JPEG image.
        timeout:     Per-request timeout in seconds.
        retries:     Total number of attempts before raising.
        retry_delay: Seconds to wait between failed attempts.
        pre_delay:   Seconds to wait before the first request (let auto-exposure
                     settle even when AE is locked — kept as a safety margin).

    Returns:
        np.ndarray: BGR image, shape (H, W, 3), dtype uint8.

    Raises:
        RuntimeError: On connection failure after all retries, or if the
                      response cannot be decoded as an image.
    """
    if pre_delay > 0:
        time.sleep(pre_delay)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()

            buf = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(
                    f"cv2.imdecode returned None — the response from {url} is not a "
                    "valid image (content-length={len(resp.content)} bytes)"
                )
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return img

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = e
            if attempt < retries:
                time.sleep(retry_delay)

        except RuntimeError:
            raise

        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"HTTP error from camera: {e}. "
                "Check that IP Webcam is running and the port forwarding is active."
            ) from e

    raise RuntimeError(
        f"Cannot connect to camera at {url} after {retries} attempt(s). "
        "Ensure the phone is connected via USB, IP Webcam is running, and "
        "'adb forward tcp:8080 tcp:8080' is active. "
        f"Last error: {last_error}"
    )
