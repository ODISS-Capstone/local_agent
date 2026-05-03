"""
ipTIME C500G 같은 ONVIF 카메라의 실제 RTSP stream URL 발급기.

카메라 펌웨어가 광고하는 path(/stream_ch0 등)가 SETUP을 거부할 때,
ONVIF GetStreamUri로 발급받은 path만 표준 RTSP 클라이언트가 통과한다.

사용법:
    python3 -m scripts.probe_onvif <ip> <onvif_port> <user> <pass>
예:
    python3 -m scripts.probe_onvif 192.168.0.75 5000 jepetolee 'neworder2030!!'

출력 후 agent_config.yaml의 rtsp.url 부분을 직접 갱신하면 된다.
"""

from __future__ import annotations

import sys
import urllib.parse

from onvif import ONVIFCamera


def main() -> None:
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    ip, port, user, pwd = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]

    cam = ONVIFCamera(ip, port, user, pwd)
    try:
        cam.update_xaddrs()
    except Exception as e:
        print(f"[warn] update_xaddrs: {e}")

    info = cam.devicemgmt.GetDeviceInformation()
    print("=== Device ===")
    for k in ("Manufacturer", "Model", "FirmwareVersion", "SerialNumber", "HardwareId"):
        print(f"  {k}: {info[k]}")

    media = cam.create_media_service()
    profiles = media.GetProfiles()
    print("\n=== Profiles ===")
    for p in profiles:
        print(f"  token={p.token}  name={p.Name}")

    pwd_enc = urllib.parse.quote(pwd, safe="")
    print("\n=== RTSP URLs (with encoded credentials) ===")
    for p in profiles:
        req = media.create_type("GetStreamUri")
        req.ProfileToken = p.token
        req.StreamSetup = {
            "Stream": "RTP-Unicast",
            "Transport": {"Protocol": "RTSP"},
        }
        uri = media.GetStreamUri(req).Uri
        # rtsp://host:port/path -> 자격 끼워넣기
        if "://" in uri:
            scheme, rest = uri.split("://", 1)
            full = f"{scheme}://{user}:{pwd_enc}@{rest}"
        else:
            full = uri
        print(f"  [{p.Name}] {full}")


if __name__ == "__main__":
    main()
