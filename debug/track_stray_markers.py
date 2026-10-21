#!/usr/bin/env python3
"""Live Polaris Vega passive stray-marker tracking, 3D plot and CSV logging.

Requires vega_client.py (PolarisVega class) in the same directory.
Uses TX 1800, which includes out-of-volume passive stray markers.
"""
import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from vega_client import PolarisVega


def parse_tx1800(response):
    """Parse the passive-stray section of an ASCII TX 1800 response.

    Expected for one enabled dummy handle: '<tool line>\n<stray line><CRC>'.
    Stray line begins with 2 hex digits for count and one status digit; each XYZ is 3 signed,
    7-character integers in 0.01 mm units. The remainder holds status and CRC.
    """
    lines = response.strip('\r').split('\n')
    if len(lines) < 2:
        raise ValueError(f'No stray-marker section in response: {response!r}')
    section = lines[-1]
    if len(section) < 2:
        raise ValueError(f'Short stray-marker section: {section!r}')
    try:
        count = int(section[:2], 16)
    except ValueError as exc:
        raise ValueError(f'Invalid stray-marker count: {section[:2]!r}') from exc
    needed = 3 + 21 * count
    if len(section) < needed + 4:
        raise ValueError(f'Truncated stray-marker section: {section!r}')
    coords = []
    for i in range(count):
        item = section[3 + i * 21: 3 + (i + 1) * 21]
        if any(item[j] not in '+-' for j in (0, 7, 14)):
            raise ValueError(f'Unexpected XYZ field format: {item!r}')
        coords.append(tuple(int(item[j:j + 7]) / 100.0 for j in (0, 7, 14)))
    return coords


def distance(a, b):
    return math.dist(a, b)


def associate(previous, detections, gate_mm):
    """Greedy gated 3D nearest-neighbor association; IDs are not permanent."""
    candidates = sorted(
        (distance(pos, det), marker_id, i)
        for marker_id, pos in previous.items()
        for i, det in enumerate(detections)
    )
    assigned, used_ids, used_detections = {}, set(), set()
    for dist, marker_id, i in candidates:
        if dist > gate_mm or marker_id in used_ids or i in used_detections:
            continue
        assigned[marker_id] = detections[i]
        used_ids.add(marker_id)
        used_detections.add(i)
    return assigned, used_detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='169.254.51.30')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--hz', type=float, default=10.0, help='Requested polling frequency')
    parser.add_argument('--gate-mm', type=float, default=80.0, help='Maximum frame-to-frame ID movement')
    parser.add_argument('--output', default=None, help='CSV output path')
    parser.add_argument('--no-plot', action='store_true', help='Log without live plotting')
    args = parser.parse_args()
    if args.hz <= 0 or args.gate_mm <= 0:
        parser.error('--hz and --gate-mm must be positive')

    output = Path(args.output or f'stray_markers_{datetime.now():%Y%m%d_%H%M%S}.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    vega = PolarisVega(host=args.host, port=args.port)
    tracking = False
    last_seen = {}  # marker ID -> last observed position
    next_id = 1
    frame = 0
    t0 = time.monotonic()

    if not args.no_plot:
        plt.ion()
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        plt.show(block=False)

    try:
        vega.connect()

        def send(cmd):
            reply = vega.command(cmd)
            if reply.startswith('ERROR'):
                raise RuntimeError(f'{cmd}: {reply}')
            return reply

        print('INIT:', send('INIT'))
        handle_reply = send('PHRQ *********1**01')
        handle = handle_reply[:2]
        if len(handle) != 2:
            raise RuntimeError(f'Unexpected PHRQ response: {handle_reply!r}')
        print('Dummy handle:', handle)
        print('PINIT:', send(f'PINIT {handle}'))
        print('PENA:', send(f'PENA {handle}D'))
        print('Enabled handles:', send('PHSR 04'))
        print('Alerts:', send('GET Info.Status.Alerts'))
        print('TSTART:', send('TSTART'))
        tracking = True
        print('Writing:', output.resolve())
        print('Press Ctrl+C to stop.')

        with output.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'time_s', 'timestamp_iso', 'marker_id', 'x_mm', 'y_mm', 'z_mm', 'in_characterized_volume'])
            period = 1.0 / args.hz
            while True:
                loop_start = time.monotonic()
                response = send('TX 1800')
                detections = parse_tx1800(response)
                now = datetime.now().astimezone().isoformat(timespec='milliseconds')
                elapsed = time.monotonic() - t0

                current, used = associate(last_seen, detections, args.gate_mm)
                for i, pos in enumerate(detections):
                    if i not in used:
                        current[next_id] = pos
                        next_id += 1
                last_seen = current  # dropped markers get new IDs on reappearance

                for marker_id, (x, y, z) in sorted(current.items()):
                    writer.writerow([frame, f'{elapsed:.6f}', now, marker_id,
                                     f'{x:.2f}', f'{y:.2f}', f'{z:.2f}', 'unknown'])
                f.flush()
                if frame % max(1, round(args.hz)) == 0:
                    print(f'Frame {frame}: {len(detections)} markers; IDs {sorted(current)}')

                if not args.no_plot:
                    ax.cla()
                    if current:
                        xs, ys, zs = zip(*current.values())
                        ax.scatter(xs, ys, zs, s=65)
                        for marker_id, (x, y, z) in current.items():
                            ax.text(x, y, z, str(marker_id))
                        # Keep all three axes in millimeters and similarly scaled.
                        cx, cy, cz = (sum(vals) / len(vals) for vals in (xs, ys, zs))
                        span = max(100.0, max(max(vals) - min(vals) for vals in (xs, ys, zs)))
                        half = span * 0.65
                        ax.set_xlim(cx - half, cx + half)
                        ax.set_ylim(cy - half, cy + half)
                        ax.set_zlim(cz - half, cz + half)
                        ax.set_box_aspect((1, 1, 1))
                    ax.set_xlabel('X (mm)')
                    ax.set_ylabel('Y (mm)')
                    ax.set_zlabel('Z (mm)')
                    ax.set_title(f'Polaris Vega stray markers | frame {frame} | {len(current)} detected')
                    fig.canvas.draw_idle()
                    plt.pause(0.001)
                    if not plt.fignum_exists(fig.number):
                        print('Plot closed; stopping.')
                        break

                frame += 1
                time.sleep(max(0.0, period - (time.monotonic() - loop_start)))
    except KeyboardInterrupt:
        print('\nStopped by user.')
    finally:
        if tracking:
            try:
                print('TSTOP:', vega.command('TSTOP'))
            except Exception as exc:
                print('TSTOP failed:', exc)
        vega.close()
        print('CSV saved to:', output.resolve())


if __name__ == '__main__':
    main()
