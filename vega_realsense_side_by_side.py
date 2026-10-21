#!/usr/bin/env python3
"""Record Vega 3D markers and RealSense RGB, with live side-by-side RGB/MuJoCo preview.

Outputs: datacollection/<timestamp>/markers.csv, rgb.mp4, mujoco.mp4,
         side_by_side.mp4, rgb_timestamps.csv, video_timestamps.csv, metadata.json.

Note: software synchronized, not hardware synchronized. MuJoCo depicts tracked
marker positions, not a physics simulation of the chopsticks.
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
import numpy as np
import pyrealsense2 as rs

from vega_client import PolarisVega

# Display transform (rows): Vega Y->MuJoCo X, -Z->MuJoCo Y, -X->MuJoCo Z.
VEGA_TO_MUJOCO = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=float)


def parse_tx1800(response):
    lines = response.replace('\r', '').split('\n')
    if len(lines) < 2:
        raise ValueError(f'Unexpected TX 1800 reply: {response!r}')
    line = lines[-1]
    count = int(line[:2], 16)
    flags_len = (count + 3) // 4
    flags = line[2:2 + flags_len]
    start = 2 + flags_len
    if len(line) != start + 21 * count + 8:
        raise ValueError(f'Unexpected TX 1800 length: {line!r}')
    points = []
    for i in range(count):
        group = line[start + 21 * i:start + 21 * (i + 1)]
        if any(group[k] not in '+-' for k in (0, 7, 14)):
            raise ValueError(f'Bad marker coordinate group: {group!r}')
        points.append(tuple(int(group[k:k + 7]) / 100.0 for k in (0, 7, 14)))
    return points, flags


def assign_ids(previous, points, next_id, max_jump_mm):
    candidates = sorted((math.dist(old, p), marker_id, j)
                        for marker_id, old in previous.items()
                        for j, p in enumerate(points))
    current, used_old, used_new = {}, set(), set()
    for dist, marker_id, j in candidates:
        if dist <= max_jump_mm and marker_id not in used_old and j not in used_new:
            current[marker_id] = points[j]
            used_old.add(marker_id)
            used_new.add(j)
    for j, p in enumerate(points):
        if j not in used_new:
            current[next_id] = p
            next_id += 1
    return current, next_id


def realsense_camera_xml(calib):
    """MuJoCo camera at the calibrated RealSense pose (see calibrate_realsense_to_vega.py)."""
    R = np.array(calib['R'])
    t = np.array(calib['t_mm'])
    pos = VEGA_TO_MUJOCO @ (-R.T @ t) / 1000
    # OpenCV camera: x right, y down, z forward. MuJoCo camera: x right, y up, looks along -z.
    x_axis = VEGA_TO_MUJOCO @ R[0]
    y_axis = VEGA_TO_MUJOCO @ -R[1]
    fovy = math.degrees(2 * math.atan(calib['height'] / (2 * calib['fy'])))
    fmt = lambda v: ' '.join(f'{c:.6f}' for c in v)
    return (f'<camera name="realsense" pos="{fmt(pos)}" '
            f'xyaxes="{fmt(x_axis)} {fmt(y_axis)}" fovy="{fovy:.4f}"/>')


def make_model(max_markers, render_width, render_height, camera_xml=''):
    colors = ['1 .25 .25 1', '.2 .8 .3 1', '.25 .5 1 1', '1 .75 .2 1',
              '.8 .3 1 1', '.2 .9 .9 1']
    sites = '\n'.join(f'<site name="marker_{i}" type="sphere" pos="0 0 -100" '
                      f'size="0.014" rgba="{colors[i % len(colors)]}"/>'
                      for i in range(max_markers))
    xml = f'''<mujoco model="Vega markers">
      <visual><global offwidth="{render_width}" offheight="{render_height}"/></visual>
      <worldbody>
        <light pos="0 1 3"/>
        {camera_xml}
        <geom type="sphere" pos="0 0 .02" size=".025" rgba=".2 .7 1 1"/>
        {sites}
      </worldbody>
    </mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)


def open_writer(path, fps, size):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f'Cannot create {path}. Check OpenCV MP4 encoder support.')
    return writer


def capture_realsense(args, stop, shared, lock, folder, start, errors):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.rgb_fps)
    started = False
    writer = None
    try:
        profile = pipeline.start(config)
        started = True
        print('RealSense:', profile.get_device().get_info(rs.camera_info.name), flush=True)
        with (folder / 'rgb_timestamps.csv').open('w', newline='') as f:
            csv_out = csv.writer(f)
            csv_out.writerow(['rgb_frame', 'host_elapsed_s', 'host_utc_iso', 'realsense_timestamp_ms'])
            index = 0
            while not stop.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1500)
                except RuntimeError:
                    if stop.is_set():
                        break
                    raise
                color = frames.get_color_frame()
                if not color:
                    continue
                frame = np.asanyarray(color.get_data()).copy()
                stamp = time.monotonic() - start
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = open_writer(folder / 'rgb.mp4', args.rgb_fps, (w, h))
                writer.write(frame)
                with lock:
                    shared['image'] = frame
                    shared['frame'] = index
                    shared['stamp'] = stamp
                csv_out.writerow([index, f'{stamp:.6f}', datetime.now(timezone.utc).isoformat(),
                                  f'{color.get_timestamp():.3f}'])
                index += 1
                if index % args.rgb_fps == 0:
                    f.flush()
    except Exception as exc:
        errors.append(repr(exc))
        stop.set()
    finally:
        if writer is not None:
            writer.release()
        if started:
            pipeline.stop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='169.254.51.30')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--vega-hz', type=float, default=30)
    p.add_argument('--rgb-fps', type=int, default=30)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--max-markers', type=int, default=16)
    p.add_argument('--gate-mm', type=float, default=70)
    p.add_argument('--data-dir', default='datacollection')
    p.add_argument('--no-preview', action='store_true', help='Record without a live preview window')
    p.add_argument('--calibration', help='camera_calibration.json; render MuJoCo from the RealSense viewpoint')
    args = p.parse_args()
    if args.vega_hz <= 0 or args.rgb_fps <= 0:
        p.error('Frame rates must be positive')
    calib = None
    if args.calibration:
        calib = json.loads(Path(args.calibration).expanduser().read_text())
        if (calib['width'], calib['height']) != (args.width, args.height):
            p.error(f"Calibration is for {calib['width']}x{calib['height']}, "
                    f'but recording at {args.width}x{args.height}')

    folder = Path(args.data_dir).expanduser() / datetime.now().strftime('%Y%m%d_%H%M%S')
    folder.mkdir(parents=True, exist_ok=False)
    print('Recording to:', folder.resolve(), flush=True)
    (folder / 'metadata.json').write_text(json.dumps({
        'vega_host': args.host, 'vega_command': 'TX 1800',
        'vega_coordinate_units': 'mm', 'vega_hz_requested': args.vega_hz,
        'rgb_fps_requested': args.rgb_fps, 'rgb_size': [args.width, args.height],
        'camera_calibration': calib,
        'note': 'Side-by-side pairs the latest RGB image with each Vega sample. '
                'Video timestamps use the host monotonic clock; not hardware-synchronized. '
                'Marker IDs are provisional; out-of-volume measurements are not accuracy-guaranteed.'
    }, indent=2))

    model = make_model(args.max_markers, args.width, args.height,
                       realsense_camera_xml(calib) if calib else '')
    data = mujoco.MjData(model)
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f'marker_{i}')
             for i in range(args.max_markers)]
    if calib:
        cam = 'realsense'
    else:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (0, 1.4, 0.05)
        cam.distance = 2.9
        cam.azimuth = 135
        cam.elevation = -20

    stop = threading.Event()
    shared, lock, errors = {'image': None, 'frame': -1, 'stamp': None}, threading.Lock(), []
    start = time.monotonic()
    rgb_thread = threading.Thread(target=capture_realsense,
                                  args=(args, stop, shared, lock, folder, start, errors),
                                  daemon=True)
    vega = PolarisVega(host=args.host, port=args.port)
    tracking = False
    render = None
    mj_writer = combo_writer = None
    rgb_thread.start()

    def send(command):
        response = vega.command(command)
        if response.startswith('ERROR'):
            raise RuntimeError(f'{command}: {response}')
        return response

    try:
        # The renderer is offscreen; it generates actual MuJoCo frames for the MP4.
        render = mujoco.Renderer(model, height=args.height, width=args.width)
        vega.connect()
        send('INIT')
        handle = send('PHRQ *********1**01')[:2]
        send(f'PINIT {handle}')
        send(f'PENA {handle}D')
        send('TSTART')
        tracking = True
        mj_writer = open_writer(folder / 'mujoco.mp4', args.vega_hz, (args.width, args.height))
        combo_writer = open_writer(folder / 'side_by_side.mp4', args.vega_hz,
                                   (args.width * 2, args.height))
        previous, next_id, index = {}, 1, 0
        with (folder / 'markers.csv').open('w', newline='') as mf, \
             (folder / 'video_timestamps.csv').open('w', newline='') as vf:
            marker_csv, video_csv = csv.writer(mf), csv.writer(vf)
            marker_csv.writerow(['vega_frame', 'host_elapsed_s', 'host_utc_iso',
                                 'marker_id', 'x_mm', 'y_mm', 'z_mm',
                                 'detected_count', 'out_of_volume_flags_raw'])
            video_csv.writerow(['video_frame', 'vega_elapsed_s', 'rgb_frame_used', 'rgb_elapsed_s'])
            print('Live preview: press q or Ctrl+C to stop', flush=True)
            while not stop.is_set():
                tick = time.monotonic()
                points, flags = parse_tx1800(send('TX 1800'))
                stamp = time.monotonic() - start
                previous, next_id = assign_ids(previous, points, next_id, args.gate_mm)
                for marker_id, (x, y, z) in sorted(previous.items()):
                    marker_csv.writerow([index, f'{stamp:.6f}', datetime.now(timezone.utc).isoformat(),
                                        marker_id, x, y, z, len(points), flags])
                if not points:
                    marker_csv.writerow([index, f'{stamp:.6f}', datetime.now(timezone.utc).isoformat(),
                                        '', '', '', '', 0, flags])
                for i, site_id in enumerate(sites):
                    if i < len(previous):
                        x, y, z = sorted(previous.items())[i][1]
                        model.site_pos[site_id] = (y / 1000, -z / 1000, -x / 1000)  # Vega Y->X, -Z->Y, -X->Z
                    else:
                        model.site_pos[site_id] = (0, 0, -100)
                mujoco.mj_forward(model, data)
                render.update_scene(data, camera=cam)
                mj_rgb = render.render()
                mj_bgr = cv2.cvtColor(mj_rgb, cv2.COLOR_RGB2BGR)
                with lock:
                    rgb = shared['image']
                    rgb_index = shared['frame']
                    rgb_stamp = shared['stamp']
                if rgb is None:
                    rgb = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                    cv2.putText(rgb, 'Waiting for RealSense...', (25, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
                else:
                    rgb = cv2.resize(rgb, (args.width, args.height))
                combined = np.hstack((rgb, mj_bgr))
                cv2.putText(combined, 'RealSense RGB', (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
                cv2.putText(combined, f'MuJoCo: {len(points)} markers', (args.width + 12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
                mj_writer.write(mj_bgr)
                combo_writer.write(combined)
                video_csv.writerow([index, f'{stamp:.6f}', rgb_index,
                                    '' if rgb_stamp is None else f'{rgb_stamp:.6f}'])
                if not args.no_preview:
                    cv2.imshow('RealSense RGB | MuJoCo (q to stop)', combined)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop.set()
                if index % max(1, round(args.vega_hz)) == 0:
                    print(f'Vega frame {index}: {len(points)} markers; RGB frame {rgb_index}', flush=True)
                    mf.flush()
                    vf.flush()
                index += 1
                time.sleep(max(0.0, 1 / args.vega_hz - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print('\nStopping recording...')
    finally:
        stop.set()
        if tracking:
            try:
                send('TSTOP')
            except Exception as exc:
                print('TSTOP warning:', exc)
        vega.close()
        if mj_writer is not None:
            mj_writer.release()
        if combo_writer is not None:
            combo_writer.release()
        if render is not None:
            render.close()
        if not args.no_preview:
            cv2.destroyAllWindows()
        rgb_thread.join(timeout=5)
        if errors:
            print('RealSense errors:', errors)
        print('Saved recordings:', folder.resolve())


if __name__ == '__main__':
    main()
