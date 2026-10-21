#!/usr/bin/env python3
"""Calibrate the RealSense RGB camera pose in the Vega frame (marker clicks + solvePnP).

Show exactly ONE marker to the Vega, hold it still, and left-click its centre in
the RGB window. Repeat at 8+ positions spread across the image AND at different
distances from the camera (non-coplanar). Keys: c = compute + save, u = undo,
q = quit. After computing, the live marker is reprojected as a green circle so
you can check the fit; you can keep adding points and press c again.

Output: camera_calibration.json with RealSense intrinsics and R, t such that
p_camera_mm = R @ p_vega_mm + t (OpenCV camera axes: x right, y down, z forward).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from vega_client import PolarisVega
from vega_realsense_side_by_side import parse_tx1800


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--host', default='169.254.51.30')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--rgb-fps', type=int, default=30)
    p.add_argument('--min-points', type=int, default=8)
    p.add_argument('--output', default='camera_calibration.json')
    args = p.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.rgb_fps)
    profile = pipeline.start(config)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    dist = np.array(intr.coeffs, dtype=float)
    print(f'RealSense intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} '
          f'cx={intr.ppx:.1f} cy={intr.ppy:.1f} model={intr.model}', flush=True)

    vega = PolarisVega(host=args.host, port=args.port)
    tracking = False
    clicks = []
    pairs = []  # (vega_xyz_mm, pixel_uv)
    pose = None
    window = 'Calibration: click the single marker (c=compute, u=undo, q=quit)'
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, lambda event, x, y, *_:
                         clicks.append((x, y)) if event == cv2.EVENT_LBUTTONDOWN else None)

    def send(command):
        response = vega.command(command)
        if response.startswith('ERROR'):
            raise RuntimeError(f'{command}: {response}')
        return response

    try:
        vega.connect()
        send('INIT')
        handle = send('PHRQ *********1**01')[:2]
        send(f'PINIT {handle}')
        send(f'PENA {handle}D')
        send('TSTART')
        tracking = True
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=1500)
            color = frames.get_color_frame()
            if not color:
                continue
            image = np.asanyarray(color.get_data()).copy()
            points, _ = parse_tx1800(send('TX 1800'))

            while clicks:
                uv = clicks.pop(0)
                if len(points) != 1:
                    print(f'Ignored click: {len(points)} markers visible, need exactly 1', flush=True)
                    continue
                pairs.append((points[0], uv))
                print(f'Point {len(pairs)}: vega={points[0]} pixel={uv}', flush=True)

            for _, (u, v) in pairs:
                cv2.drawMarker(image, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
            if pose is not None and points:
                projected, _ = cv2.projectPoints(np.array(points), pose[0], pose[1], K, dist)
                for u, v in projected.reshape(-1, 2):
                    cv2.circle(image, (int(round(u)), int(round(v))), 10, (0, 255, 0), 2)
            cv2.putText(image, f'{len(points)} markers visible | {len(pairs)} points',
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            cv2.imshow(window, image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('u') and pairs:
                pairs.pop()
                print(f'Removed last point; {len(pairs)} left', flush=True)
            if key == ord('c'):
                if len(pairs) < args.min_points:
                    print(f'Need at least {args.min_points} points, have {len(pairs)}', flush=True)
                    continue
                obj = np.array([xyz for xyz, _ in pairs], dtype=float)
                img = np.array([uv for _, uv in pairs], dtype=float)
                # SQPnP is globally optimal; ITERATIVE can land on a mirrored pose behind the camera.
                ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
                if not ok:
                    print('solvePnP failed; add more spread-out points', flush=True)
                    continue
                rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
                depths = (obj @ cv2.Rodrigues(rvec)[0].T + tvec.ravel())[:, 2]
                if (depths <= 0).any():
                    print('Solution puts points behind the camera; add more spread-out points', flush=True)
                    continue
                projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
                errors = np.linalg.norm(projected.reshape(-1, 2) - img, axis=1)
                rms = float(np.sqrt(np.mean(errors ** 2)))
                pose = (rvec, tvec)
                R = cv2.Rodrigues(rvec)[0]
                Path(args.output).write_text(json.dumps({
                    'width': args.width, 'height': args.height,
                    'fx': intr.fx, 'fy': intr.fy, 'cx': intr.ppx, 'cy': intr.ppy,
                    'dist_coeffs': dist.tolist(),
                    'R': R.tolist(), 't_mm': tvec.ravel().tolist(),
                    'rms_reprojection_px': rms,
                    'per_point_error_px': errors.tolist(),
                    'pairs': [{'vega_mm': list(xyz), 'pixel': list(uv)} for xyz, uv in pairs],
                }, indent=2))
                print(f'Saved {args.output}: RMS reprojection error {rms:.2f}px '
                      f'(worst {errors.max():.2f}px, point {int(errors.argmax()) + 1})', flush=True)
    finally:
        if tracking:
            try:
                send('TSTOP')
            except Exception as exc:
                print('TSTOP warning:', exc)
        vega.close()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
