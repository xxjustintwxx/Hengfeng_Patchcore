"""
Capture a single snapshot from a camera — either an Android phone running the
IP Webcam app (HTTP JPEG endpoint) or a USB/UVC webcam plugged into this
machine (via OpenCV).

Usage:
    from capture import capture_frame
    bgr = capture_frame(cfg["camera"])  # returns H×W×3 BGR numpy array, per
                                         # cfg["camera"]["type"] ("http" or "usb")
"""
import time

import cv2
import numpy as np
import requests

_ROTATIONS = {
    None: None,
    "none": None,
    "90_cw": cv2.ROTATE_90_CLOCKWISE,
    "90_ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}


def _apply_rotation(img: np.ndarray, rotate) -> np.ndarray:
    if rotate not in _ROTATIONS:
        raise RuntimeError(
            f"Unknown camera.rotate value: {rotate!r} "
            f"(expected one of {sorted(k for k in _ROTATIONS if k)})"
        )
    code = _ROTATIONS[rotate]
    return cv2.rotate(img, code) if code is not None else img


def capture_snapshot(
    url: str = "http://localhost:8080/shot.jpg",
    timeout: float = 3.0,
    retries: int = 3,
    retry_delay: float = 0.5,
    pre_delay: float = 0.3,
    rotate: str = "90_cw",
) -> np.ndarray:
    """Fetch one JPEG frame from IP Webcam and return it as a BGR numpy array.

    Args:
        url:         HTTP endpoint that returns a single JPEG image.
        timeout:     Per-request timeout in seconds.
        retries:     Total number of attempts before raising.
        retry_delay: Seconds to wait between failed attempts.
        pre_delay:   Seconds to wait before the first request (let auto-exposure
                     settle even when AE is locked — kept as a safety margin).
        rotate:      One of None/"none", "90_cw", "90_ccw", "180". Defaults to
                     "90_cw" since phones are normally held in portrait for
                     this rig while IP Webcam reports the sensor's landscape
                     frame.

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
            return _apply_rotation(img, rotate)

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


def capture_snapshot_usb(
    index: int = 0,
    warmup_frames: int = 5,
    width: int | None = None,
    height: int | None = None,
    retries: int = 3,
    retry_delay: float = 0.5,
    rotate: str | None = None,
) -> np.ndarray:
    """Grab one frame from a USB/UVC webcam via OpenCV.

    Args:
        index:         OpenCV camera index (0, 1, ... — order isn't guaranteed
                        stable across reboots if more than one camera is
                        attached; check with the Camera app or Device Manager
                        if unsure which index this webcam ends up at).
        warmup_frames:  Frames to read and discard right after opening, so the
                        webcam's auto-exposure/auto-white-balance settle
                        before the frame that's kept (same role as
                        capture_snapshot's pre_delay for the phone).
        width, height:  Optional capture resolution request in pixels. Left at
                        the device's default when None.
        retries:        Total attempts (each re-opens the device) before
                        raising.
        retry_delay:    Seconds to wait between failed attempts.
        rotate:         One of None/"none", "90_cw", "90_ccw", "180".

    Returns:
        np.ndarray: BGR image, shape (H, W, 3), dtype uint8.

    Raises:
        RuntimeError: if the device can't be opened or no frame can be read
                      after all retries.
    """
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        # CAP_DSHOW: MSMF (OpenCV's default backend on Windows) can take
        # several seconds to open a UVC device; DirectShow opens the same
        # cameras near-instantly.
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                last_error = RuntimeError(f"Cannot open USB camera at index {index}")
            else:
                if width:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                if height:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                for _ in range(warmup_frames):
                    cap.read()
                ok, frame = cap.read()
                if not ok or frame is None:
                    last_error = RuntimeError(
                        f"Failed to read a frame from USB camera at index {index}"
                    )
                else:
                    return _apply_rotation(frame, rotate)
        finally:
            cap.release()
        if attempt < retries:
            time.sleep(retry_delay)

    raise RuntimeError(
        f"Cannot capture from USB camera at index {index} after {retries} "
        "attempt(s). Check that it's plugged in, powered, and not already "
        f"held open by another app (e.g. the Camera app, a video call). Last error: {last_error}"
    )


def capture_frame(cam: dict) -> np.ndarray:
    """Dispatch to the phone (HTTP) or USB webcam capture path.

    cam["type"] selects the path — "http" (default, IP Webcam over HTTP) or
    "usb" (local UVC webcam via OpenCV) — with the rest of cam read as that
    path's keyword args. This is what app.py and live_infer.py should call;
    it keeps them from needing to know which camera backend a profile uses.
    """
    cam_type = cam.get("type", "http")
    if cam_type == "http":
        return capture_snapshot(
            url=cam["url"],
            timeout=cam.get("timeout", 3.0),
            retries=cam.get("retries", 3),
            retry_delay=cam.get("retry_delay", 0.5),
            pre_delay=cam.get("pre_delay", 0.3),
            rotate=cam.get("rotate", "90_cw"),
        )
    elif cam_type == "usb":
        return capture_snapshot_usb(
            index=cam.get("index", 0),
            warmup_frames=cam.get("warmup_frames", 5),
            width=cam.get("width"),
            height=cam.get("height"),
            retries=cam.get("retries", 3),
            retry_delay=cam.get("retry_delay", 0.5),
            rotate=cam.get("rotate"),
        )
    else:
        raise RuntimeError(f"Unknown camera.type: {cam_type!r} (expected 'http' or 'usb')")
