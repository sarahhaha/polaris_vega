#!/usr/bin/env python3
"""Polaris Vega passive stray markers + Intel RealSense RGB + MuJoCo.

Outputs are placed under datacollection/<session_timestamp>/:
  markers.csv, rgb.mp4, rgb_timestamps.csv, metadata.json

RGB frames and Vega samples are independently timestamped using the same host
monotonic clock; video playback FPS is nominal, not a synchronization guarantee.
"""
import argparse
import csv
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import pyrealsense2 as rs

from vega_client import PolarisVega


def parse_tx1800(response):
    """Decode the observed NDI TX 1800 ASCII stray-marker format.

    Line 2: 2 hex count, ceil(count/4) hex out-of-volume flags,
    count * (signed 7-char X/Y/Z in 0.01 mm), 4-char status, 4-char CRC.
    Returns points in mm and raw flag field. Assumes firmware response format
    observed on this Vega ST; fails loudly on unexpected layouts.
    """
    lines = response.replace('\r', '').split('\n')
    if len(lines) < 2:
        raise ValueError(f'TX 1800 missing stray-marker line: {response!r}')
    line = lines[-1]
    if len(line) < 10:
        raise ValueError(f'TX 1800 too short: {response!r}')
    count = int(line[:2], 16)
    flag_width = (count + 3) // 4
    flags = line[2:2 + flag_width]
    start = 2 + flag_width
    end = start + count * 21
    if len(line) != end + 8:
        raise ValueError(f'Unexpected TX 1800 length: count={count}, line={line!r}')
    points = []
    for i in range(count):
        part = line[start + i * 21:start + (i + 1) * 21]
        if any(part[k] not in '+-' for k in (0, 7, 14)):
            raise ValueError(f'Invalid coordinate group {part!r}')
        points.append(tuple(int(part[k:k + 7]) / 100.0 for k in (0, 7, 14)))
    return points, flags


def assign_ids(previous, points, next_id, gate_mm):
    """Greedy one-to-one nearest-neighbor IDs; not guaranteed across crossings."""
    pairs = sorted((math.dist(old, new), marker_id, j)
                   for marker_id, old in previous.items()
                   for j, new in enumerate(points))
    current, used_ids, used_j = {}, set(), set()
    for distance, marker_id, j in pairs:
        if distance <= gate_mm and marker_id not in used_ids and j not in used_j:
            current[marker_id] = points[j]
            used_ids.add(marker_id)
            used_j.add(j)
    for j, point in enumerate(points):
        if j not in used_j:
            current[next_id] = point
            next_id += 1
    return current, next_id


def make_model(max_markers):
    sites = '\n'.join(
        f'<site name="marker_{i}" type="sphere" pos="0 0 -100" '
        f'size="0.013" rgba="{(i * 0.37) % 1:.3f} 0.75 0.35 1"/>'
        for i in range(max_markers)
    )
    xml = f'''<mujoco model="Vega marker tracking">
      <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="3 3 0.1" rgba="0.22 0.24 0.27 1"/>
        <geom type="sphere" pos="0 0 0.04" size="0.025" rgba="0.2 0.65 1 1"/>
        {sites}
      </worldbody>
    </mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)


def capture_rgb(args, session_dir, stop_event, start_monotonic, errors):
    """Capture RealSense RGB independently of Vega polling and MuJoCo rendering."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.rgb_fps)
    started = False
    writer = None
    try:
        profile = pipeline.start(config)
        started = True
        device = profile.get_device()
        print('RealSense:', device.get_info(rs.camera_info.name), flush=True)
        with (session_dir / 'rgb_timestamps.csv').open('w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['rgb_frame', 'host_elapsed_s', 'host_utc_iso', 'realsense_timestamp_ms'])
            frame_number = 0
            while not stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1500)
                except RuntimeError as exc:
                    if stop_event.is_set():
                        break
                    raise RuntimeError(f'RealSense frame timeout/error: {exc}') from exc
                color = frames.get_color_frame()
                if not color:
                    continue
                now_mono = time.monotonic()
                utc = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
                img = np.asanyarray(color.get_data())
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(str(session_dir / 'rgb.mp4'),
                                             cv2.VideoWriter_fourcc(*'mp4v'),
                                             args.rgb_fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError('Failed to open rgb.mp4 video writer')
                writer.write(img)
                csv_writer.writerow([frame_number, f'{now_mono - start_monotonic:.6f}',
                                     utc, f'{color.get_timestamp():.3f}'])
                if args.show_rgb:
                    preview = img.copy()
                    cv2.putText(preview, f'RGB frame {frame_number}', (12, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('RealSense RGB (press q to stop)', preview)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                frame_number += 1
                if frame_number % args.rgb_fps == 0:
                    f.flush()
    except Exception as exc:
        errors.append(str(exc))
        stop_event.set()
    finally:
        if writer is not None:
            writer.release()
        if started:
            pipeline.stop()
        if args.show_rgb:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='169.254.51.30')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--vega-hz', type=float, default=30)
    parser.add_argument('--rgb-fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--max-markers', type=int, default=16)
    parser.add_argument('--gate-mm', type=float, default=70)
    parser.add_argument('--data-dir', default='datacollection')
    parser.add_argument('--show-rgb', action='store_true', help='Show an RGB preview window')
    parser.add_argument('--no-mujoco', action='store_true', help='Record without the MuJoCo window')
    args = parser.parse_args()
    if args.vega_hz <= 0 or args.rgb_fps <= 0 or args.max_markers <= 0:
        parser.error('Rates and max-markers must be positive')

    session_dir = Path(args.data_dir).expanduser() / datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / 'metadata.json').write_text(json.dumps({
        'camera': 'Intel RealSense color', 'vega_host': args.host,
        'vega_command': 'TX 1800', 'vega_units': 'millimeters',
        'vega_hz_requested': args.vega_hz, 'rgb_fps_requested': args.rgb_fps,
        'rgb_size': [args.width, args.height],
        'note': 'Vega coordinates are camera-frame; markers may be outside characterized volume. '
                'Nearest-neighbor marker IDs are provisional. Video FPS is nominal; use timestamp CSVs.'
    }, indent=2))
    print('Recording folder:', session_dir.resolve(), flush=True)

    model = make_model(args.max_markers) if not args.no_mujoco else None
    data = mujoco.MjData(model) if model is not None else None
    site_ids = ([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f'marker_{i}')
                 for i in range(args.max_markers)] if model is not None else [])

    stop_event = threading.Event()
    errors = []
    start_monotonic = time.monotonic()
    rgb_thread = threading.Thread(target=capture_rgb,
                                  args=(args, session_dir, stop_event, start_monotonic, errors),
                                  daemon=True)
    vega = PolarisVega(host=args.host, port=args.port)
    tracking = False
    rgb_thread.start()

    def send(command):
        reply = vega.command(command)
        if reply.startswith('ERROR'):
            raise RuntimeError(f'{command} -> {reply}')
        return reply

    def track_loop(viewer, csv_writer, file_handle):
        nonlocal tracking
        previous, next_id, frame = {}, 1, 0
        interval = 1.0 / args.vega_hz
        while not stop_event.is_set() and (viewer is None or viewer.is_running()):
            tick = time.monotonic()
            response = send('TX 1800')
            now = time.monotonic()
            points, flags = parse_tx1800(response)
            current, next_id = assign_ids(previous, points, next_id, args.gate_mm)
            previous = current
            utc = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
            if not current:
                csv_writer.writerow([frame, f'{now - start_monotonic:.6f}', utc, '', '', '', '', 0, flags])
            for marker_id, (x, y, z) in sorted(current.items()):
                csv_writer.writerow([frame, f'{now - start_monotonic:.6f}', utc,
                                     marker_id, x, y, z, len(points), flags])
            if viewer is not None:
                with viewer.lock():
                    for i, site_id in enumerate(site_ids):
                        if i < len(current):
                            x, y, z = list(sorted(current.items()))[i][1]
                            # Display transform: Vega Y->MuJoCo X, -Z->MuJoCo Y, -X->MuJoCo Z.
                            model.site_pos[site_id] = (y / 1000, -z / 1000, -x / 1000)
                        else:
                            model.site_pos[site_id] = (0, 0, -100)
                    mujoco.mj_forward(model, data)
                viewer.sync()
            if frame % max(1, round(args.vega_hz)) == 0:
                print(f'Vega frame {frame}: {len(points)} markers | RGB recording active', flush=True)
                file_handle.flush()
            frame += 1
            time.sleep(max(0.0, interval - (time.monotonic() - tick)))

    try:
        vega.connect()
        send('INIT')
        handle = send('PHRQ *********1**01')[:2]
        send(f'PINIT {handle}')
        send(f'PENA {handle}D')
        send('TSTART')
        tracking = True
        with (session_dir / 'markers.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['vega_frame', 'host_elapsed_s', 'host_utc_iso', 'marker_id',
                             'x_mm', 'y_mm', 'z_mm', 'detected_count', 'out_of_volume_flags_raw'])
            if args.no_mujoco:
                track_loop(None, writer, f)
            else:
                with mujoco.viewer.launch_passive(model, data) as viewer:
                    viewer.cam.lookat[:] = (0, 1.3, 0)
                    viewer.cam.distance = 3.1
                    viewer.cam.azimuth = 120
                    viewer.cam.elevation = -25
                    print('Close MuJoCo viewer, press Ctrl+C, or press q in RGB preview to stop.')
                    track_loop(viewer, writer, f)
    except KeyboardInterrupt:
        print('\nStopping on Ctrl+C')
    finally:
        stop_event.set()
        if tracking:
            try:
                print('TSTOP:', vega.command('TSTOP'))
            except Exception as exc:
                print('TSTOP error:', exc)
        vega.close()
        rgb_thread.join(timeout=5)
        if rgb_thread.is_alive():
            print('WARNING: RGB capture did not stop within 5 seconds')
        if errors:
            print('RGB capture errors:', errors)
        print('Saved data under:', session_dir.resolve())


if __name__ == '__main__':
    main()
