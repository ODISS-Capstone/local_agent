"""ipTIME 카메라 RTSP URL 후보를 빠르게 시험한다.

사용법:
    python3 -m scripts.probe_rtsp [user] [password]

- 인자 생략 시 비인증으로만 시도.
- 사용자/비번을 주면 user:pw@ip 형식으로도 시도.
"""

from __future__ import annotations

import os
import sys
import urllib.parse

import cv2

IP = os.environ.get("CAM_IP", "192.168.0.75")
PORT = os.environ.get("CAM_PORT", "554")

# ipTIME / 일반 IP카메라 흔한 RTSP 경로
CANDIDATE_PATHS = [
    "live/ch00_0",
    "live/ch01_0",
    "live/main",
    "live/sub",
    "11",
    "12",
    "stream1",
    "stream0",
    "video.h264",
    "cam/realmonitor?channel=1&subtype=0",
    "cam/realmonitor?channel=1&subtype=1",
    "Streaming/Channels/101",
    "Streaming/Channels/102",
    "h264",
    "0",
    "1",
]


def build_url(path: str, user: str | None, password: str | None) -> str:
    if user:
        u = urllib.parse.quote(user, safe="")
        p = urllib.parse.quote(password or "", safe="")
        cred = f"{u}:{p}@"
    else:
        cred = ""
    return f"rtsp://{cred}{IP}:{PORT}/{path}"


def try_url(url: str, timeout_ms: int = 4000) -> bool:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;tcp|stimeout;{timeout_ms * 1000}"
    )
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return False
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    h, w = frame.shape[:2]
    print(f"    -> OK  {w}x{h}")
    return True


def main() -> None:
    user = sys.argv[1] if len(sys.argv) > 1 else None
    password = sys.argv[2] if len(sys.argv) > 2 else None

    if user:
        print(f"[probe] target={IP}:{PORT}, user={user}")
    else:
        print(f"[probe] target={IP}:{PORT}, no auth")

    found: list[str] = []
    for path in CANDIDATE_PATHS:
        url = build_url(path, user, password)
        print(f"[try] {url}")
        try:
            if try_url(url):
                found.append(url)
        except Exception as e:
            print(f"    -> err {e.__class__.__name__}: {e}")

    print()
    if found:
        print("[probe] === 사용 가능한 URL ===")
        for u in found:
            print(f"  {u}")
    else:
        print("[probe] 사용 가능한 URL 없음. 인증 정보가 필요하거나 경로가 다를 수 있음.")


if __name__ == "__main__":
    main()
