# Polaris Vega + RealSense + MuJoCo

Record passive-marker positions from an NDI Polaris Vega together with Intel RealSense RGB video, and visualize the markers in MuJoCo, optionally rendered from the RealSense camera's point of view.

The recordings are the input for [chopsticks](https://github.com/sarahhaha/chopsticks), which labels the markers and estimates the chopstick tips.

## Setup

Tested with Python 3.11, numpy 2.4, OpenCV 5.0, MuJoCo 3.16, and pyrealsense2 2.59.

```bash
conda create -n polaris-vega python=3.11
conda activate polaris-vega
pip install numpy opencv-python mujoco pyrealsense2
```

Before you run anything:

- The Vega must be reachable over Ethernet. All scripts default to `--host 169.254.51.30 --port 8765`.
- The RealSense must be on USB 3.

## Workflow

### 1. Calibrate the RealSense against the Vega (once per camera setup)

```bash
python calibrate_realsense_to_vega.py
```

1. Show **exactly one** marker to the Vega and hold it still.
2. Left-click the marker's centre in the RGB window.
3. Repeat at 8 or more positions. Spread them across the image **and** at different distances from the camera, so the points don't all lie in one plane.
4. Press `c` to compute. The script saves `camera_calibration.json` and prints the RMS reprojection error, which should be a few pixels at most.
5. Check the green circle: it shows where the calibration predicts the live marker. If the fit is poor, press `u` to undo bad points, add more, and press `c` again.
6. Press `q` to quit.

Recalibrate whenever either camera moves. The calibration is only valid at the resolution it was made at (default 640×480).

### 2. Record

```bash
python vega_realsense_side_by_side.py --calibration camera_calibration.json
```

- **Countdown:** a live preview runs for 5 seconds before anything is saved. Change it with `--countdown N`, or skip it with `--countdown 0`.
- **Preview:** the RealSense RGB image appears on the left and the MuJoCo rendering of the markers on the right.
- **Camera view:** with `--calibration`, MuJoCo renders from the RealSense pose so the two halves line up. Without it, MuJoCo uses a fixed overview camera.
- **Stopping:** press `q` or `Ctrl+C`.

Each run creates `datacollection/<YYYYmmdd_HHMMSS>/`. All elapsed times count from the moment recording starts.

| File | Contents |
|---|---|
| `markers.csv` | One row per marker per Vega frame: `vega_frame, host_elapsed_s, host_utc_iso, marker_id, x_mm, y_mm, z_mm, detected_count, out_of_volume_flags_raw` |
| `rgb.mp4`, `rgb_timestamps.csv` | Raw RealSense video and per-frame host and RealSense timestamps |
| `mujoco.mp4` | MuJoCo rendering, one frame per Vega sample |
| `side_by_side.mp4` | RGB and MuJoCo side by side |
| `video_timestamps.csv` | Which RGB frame was paired with each Vega frame |
| `metadata.json` | Settings, plus the camera calibration if one was used |

Useful options: `--vega-hz`, `--rgb-fps`, `--width/--height`, `--max-markers`, `--gate-mm` (the maximum jump allowed per frame for a marker to keep its ID), `--no-preview`.

## Other scripts

- `vega_realsense_mujoco.py` is the older recorder. It uses an interactive MuJoCo viewer and has no side-by-side video, countdown, or calibration.
- `vega_client.py` is a minimal TCP client for the Vega's ASCII API.
- `debug/` holds one-off diagnostics: API version checks, tool status, a reset after a crashed session (`reset_vega.py`), and tracking tests. Run them from the repo root so they can import `vega_client`:
  ```bash
  PYTHONPATH=. python debug/reset_vega.py
  ```

## Coordinate frames

- **Vega:** reports marker positions in millimetres, in its own frame, with the origin at the tracker.
- **Vega to MuJoCo:** positions are converted to metres and the axes are mapped as **Vega Y → MuJoCo X, −Z → Y, −X → Z** (`VEGA_TO_MUJOCO` in `vega_realsense_side_by_side.py`). The blue sphere at the MuJoCo origin marks the Vega.
- **Calibration:** `camera_calibration.json` stores `R` and `t_mm`, which convert a Vega point into the RealSense frame: `p_cam = R · p_vega + t`, in OpenCV camera axes.

## Limitations

- **Synchronization:** RGB and Vega samples are matched in software on the host clock, not hardware-synchronized. Fast motion can be off by a frame or two.
- **Marker IDs:** IDs come from nearest-neighbour tracking only. They can swap when markers cross or drop out.
- **Out-of-volume markers:** markers outside the Vega's measurement volume are flagged in `out_of_volume_flags_raw`, and their accuracy is not guaranteed.
- **Rendering mismatch:** the MuJoCo camera ignores lens distortion and principal-point offset, so expect a few pixels of mismatch near the image edges.
