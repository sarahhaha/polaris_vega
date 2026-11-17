# Polaris Vega + RealSense + MuJoCo

Record passive-marker positions from an NDI Polaris Vega together with Intel RealSense RGB video (and, optionally, the RGB camera of Project Aria Gen 2 glasses), and visualize the markers in MuJoCo, optionally rendered from the RealSense camera's point of view.

The recordings are the input for [chopsticks](https://github.com/sarahhaha/chopsticks), which labels the markers and estimates the chopstick tips.

## Setup

Tested with Python 3.11, numpy 2.4, OpenCV 5.0, MuJoCo 3.16, and pyrealsense2 2.59.

```bash
conda create -n polaris-vega python=3.11
conda activate polaris-vega
pip install numpy opencv-python mujoco pyrealsense2
pip install projectaria-client-sdk==2.5.1   # only for --aria
```

Before you run anything:

- The Vega must be reachable over Ethernet. All scripts default to `--host 169.254.51.30 --port 8765`.
- The RealSense must be on USB 3.
- For `--aria`: the glasses must be paired with this computer, and their streaming certificates installed once:
  ```bash
  aria_gen2 streaming install-certs
  ```
  Run it again if streaming fails with `couldn't read cert file`.

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

Each run creates `<data folder>/<YYYYmmdd_HHMMSS>/`. All elapsed times count from the moment recording starts.

**Data folder:** `/media/nayoung/Extreme/chopsticks_data` (the external drive) by default, set in `data_paths.py`. chopsticks reads sessions from the same place. Override it with `--data-dir DIR`, or set `CHOPSTICKS_DATA` in your shell, which changes it for both repos. If the drive isn't mounted, the recorder stops with an error instead of saving somewhere else.

The drive is exFAT, which doesn't allow `:` in filenames or symlinks. Nothing here uses either, except the Aria SDK's VRS filenames: `--aria-vrs` records to `~/.cache/polaris_vega/aria_vrs/` and moves the file into the session when recording stops, with `:` replaced by `-`.

| File | Contents |
|---|---|
| `markers.csv` | One row per marker per Vega frame: `vega_frame, host_elapsed_s, host_utc_iso, marker_id, x_mm, y_mm, z_mm, detected_count, out_of_volume_flags_raw` |
| `rgb.mp4`, `rgb_timestamps.csv` | Raw RealSense video and per-frame host and RealSense timestamps |
| `mujoco.mp4` | MuJoCo rendering, one frame per Vega sample |
| `side_by_side.mp4` | RGB and MuJoCo side by side |
| `video_timestamps.csv` | Which RGB frame was paired with each Vega frame |
| `metadata.json` | Settings, plus the camera calibration if one was used |

#### Aria Gen 2 glasses

```bash
python vega_realsense_side_by_side.py --calibration camera_calibration.json --aria
```

`--aria` streams the glasses' RGB camera over USB and adds it as a third panel in the preview and in `side_by_side.mp4`. It also writes:

| File | Contents |
|---|---|
| `aria_rgb.mp4` | Aria RGB video at full resolution (2016×1512 at 10 fps with the default profile) |
| `aria_rgb_timestamps.csv` | Per frame: `aria_frame, host_elapsed_s` (arrival on the host), `host_utc_iso, aria_capture_ns` (device clock), `capture_host_elapsed_s` (capture time on the recording clock), `device_minus_host_ns` |
| `aria_vrs/*.vrs` | With `--aria-vrs`: the full stream (RGB, 4 SLAM cameras, IMUs, eye gaze, hand tracking, VIO, ...), readable with `projectaria_tools`. It covers the countdown too and uses the device clock. |

`video_timestamps.csv` gains `aria_frame_used, aria_elapsed_s`, and `metadata.json` gains an `aria` entry with the serial, profile and clock offset. `capture_host_elapsed_s` converts the device capture time to the host clock using the SDK's time sync, so it removes the ~50 ms streaming latency that `host_elapsed_s` includes.

Aria options: `--aria-ip` (if auto-discovery fails; over USB, `ip neigh` lists it on the interface named after the glasses' serial), `--aria-profile` (`mp_streaming_demo`, the default, gives 2016×1512 at 10 fps; `profile9` gives 2560×1920 at 5 fps; set `--aria-fps` to match), `--aria-interface usb|wifi_sta|wifi_sap`, `--aria-vrs`.

Over Wi-Fi: `--aria --aria-interface wifi_sta --aria-ip <Wi-Fi IP>` (the `Wi-Fi IP address` in `aria_gen2 device status`). Unplugged, the glasses turn Wi-Fi off unless Wi-Fi keep-on is set, and a restart resets it. The recorder sets keep-on every time it connects (it prints `Aria: Wi-Fi keep-on set`), so after a restart, run it once with the glasses plugged in over USB, using either interface, before you unplug. By hand: `aria_gen2 device wifi keep-on --set 1`. Without USB, `aria_gen2` commands need `--device-ip <Wi-Fi IP>`.

Useful options: `--vega-hz`, `--rgb-fps`, `--width/--height`, `--max-markers`, `--gate-mm` (the maximum jump allowed per frame for a marker to keep its ID), `--no-preview`.

## Other scripts

- `vega_realsense_mujoco.py` is the older recorder. It uses an interactive MuJoCo viewer and has no side-by-side video, countdown, or calibration.
- `aria_capture.py` holds the Aria streaming used by `--aria`.
- `data_paths.py` sets where sessions are saved.
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

- **Synchronization:** RGB and Vega samples are matched in software on the host clock, not hardware-synchronized. Fast motion can be off by a frame or two. Aria frames are paired using their capture time (`capture_host_elapsed_s`), but the Aria RGB camera only runs at 10 fps.
- **Marker IDs:** IDs come from nearest-neighbour tracking only. They can swap when markers cross or drop out.
- **Out-of-volume markers:** markers outside the Vega's measurement volume are flagged in `out_of_volume_flags_raw`, and their accuracy is not guaranteed.
- **Rendering mismatch:** the MuJoCo camera ignores lens distortion and principal-point offset, so expect a few pixels of mismatch near the image edges.
