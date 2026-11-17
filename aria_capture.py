"""Stream the RGB camera of Project Aria Gen 2 glasses alongside the Vega recording.

Used by vega_realsense_side_by_side.py --aria. Needs projectaria-client-sdk (Gen 2) and
streaming certificates installed once with `aria_gen2 streaming install-certs`.
"""
import csv
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as stream_receiver

CERT_DIR = Path.home() / '.aria' / 'streaming-certs' / 'persistent'
# The SDK names VRS files with ':' (2026-11-16-15:51:39.vrs), which exFAT drives reject, so VRS is
# recorded here first and moved into the session folder (':' -> '-') when streaming stops.
VRS_STAGING = Path.home() / '.cache' / 'polaris_vega' / 'aria_vrs'
INTERFACES = {
    'usb': sdk_gen2.StreamingInterface.USB_NCM,
    'wifi_sta': sdk_gen2.StreamingInterface.WIFI_STA,
    'wifi_sap': sdk_gen2.StreamingInterface.WIFI_SAP,
}


class AriaCapture:
    """Streams Aria RGB into shared['aria_*'] for the preview, and records while `recording` is set.

    Writes aria_rgb.mp4 and aria_rgb_timestamps.csv (plus aria_vrs/<time>.vrs with every sensor if vrs=True).
    Elapsed times use the same host monotonic clock and start time (shared['start']) as the rest
    of the recording.
    """

    def __init__(self, shared, lock, recording, ip=None, profile='mp_streaming_demo',
                 interface='usb', port=6768, fps=10, vrs=False):
        self.folder = None
        self.shared, self.lock, self.recording = shared, lock, recording
        self.ip, self.profile, self.interface, self.port, self.fps, self.vrs = (
            ip, profile, interface, port, fps, vrs)
        self.client = self.device = self.receiver = None
        self.streaming = False
        self.writer = None
        self.csv_file = self.csv_out = None
        self.index = 0
        self.frame_size = None
        self.closed = False
        self.vrs_staging = None
        self.stopping = False
        self.write_lock = threading.Lock()
        # Held by callbacks while they use the SDK; close() takes it to set `stopping`, so no callback
        # touches the receiver while stop_server() tears it down (that segfaults).
        self.sdk_lock = threading.Lock()

    def connect(self):
        """Connect to the glasses and start streaming (preview only until start() gives a folder).

        Call before creating the session folder, so glasses that can't stream leave no folder behind.
        """
        if not (CERT_DIR / 'publisher-cert-name').is_file():
            raise RuntimeError(f'No Aria streaming certificates in {CERT_DIR}. '
                               'Run once: aria_gen2 streaming install-certs')
        self.keep_wifi_on()
        self.client = sdk_gen2.DeviceClient()
        self.client.set_client_config(sdk_gen2.DeviceClientConfig())
        target = sdk_gen2.DeviceTarget(ip=self.ip) if self.ip else sdk_gen2.DeviceTarget()
        try:
            self.device = self.client.connect(target)
        except Exception as exc:
            where = f' at {self.ip}' if self.ip else ''
            if self.interface == 'usb':
                hint = ('  Check that they are switched on, awake, and plugged in over USB\n'
                        '  (`lsusb` should list "Aria Gen 2"). If lsusb lists them but this still fails,\n'
                        '  pass --aria-ip <ip>: `ip neigh` shows it on the network interface named after\n'
                        '  the glasses serial.')
            else:
                hint = ('  Check that they are switched on, awake, and on the same Wi-Fi network as this\n'
                        '  computer, and that --aria-ip is their current Wi-Fi address. With the glasses\n'
                        '  on USB, `aria_gen2 device status` shows it (wifi_ip_address).')
            raise RuntimeError(f'Cannot connect to the Aria glasses{where}: {exc}\n{hint}') from None
        print('Aria:', self.device.serial(), flush=True)

        cert_name_file = CERT_DIR / 'publisher-cert-name'
        config = sdk_gen2.HttpStreamingConfig()
        config.profile_name = self.profile
        config.streaming_interface = INTERFACES[self.interface]
        # Use the installed certificates. Left empty, the SDK generates new ones on the fly and the
        # receiver can read them half-written.
        config.streaming_cert_name = cert_name_file.read_text().strip()
        self.device.set_streaming_config(config)

        server_config = sdk_gen2.HttpServerConfig()
        server_config.address = '0.0.0.0'
        server_config.port = self.port
        self.receiver = stream_receiver.StreamReceiver(enable_image_decoding=True)
        self.receiver.set_server_config(server_config)
        if self.vrs:
            self.vrs_staging = VRS_STAGING / datetime.now().strftime('%Y%m%d_%H%M%S')
            self.vrs_staging.mkdir(parents=True, exist_ok=True)
            self.receiver.record_to_vrs(str(self.vrs_staging))  # a folder; the SDK names the file
        self.receiver.register_rgb_callback(self._on_rgb)
        self.receiver.start_server()
        try:
            self.device.start_streaming()
        except RuntimeError as exc:
            self.close()
            if 'sleep' in str(exc).lower():
                hint = 'Wake them (unfold the arms and put them on, or press a button), then run again.'
            elif 'not available' in str(exc).lower() or 'initializationerror' in str(exc).lower():
                hint = ('Their sensors failed to start. Restart the glasses, wait until '
                        '`aria_gen2 device status` shows "Device state: Ready", then run again.')
            else:
                hint = 'Check `aria_gen2 device status`, then run again.'
            raise RuntimeError(f'The Aria glasses could not start streaming: {exc}\n  {hint}') from None
        self.streaming = True

    def keep_wifi_on(self):
        """Tell the glasses to keep Wi-Fi on when unplugged (they turn it off otherwise, and a restart
        resets the setting). The Python SDK has no call for this, so it uses the aria_gen2 CLI.
        Failure only warns: USB streaming doesn't need it, and over Wi-Fi the connect error says more."""
        cli = shutil.which('aria_gen2')
        if cli is None:
            print('Aria: aria_gen2 CLI not found, cannot set Wi-Fi keep-on', flush=True)
            return
        cmd = [cli] + (['--device-ip', self.ip] if self.ip else []) + ['device', 'wifi', 'keep-on', '--set', '1']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            ok = result.returncode == 0 and 'Successfully' in result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            ok, result = False, None
        if ok:
            print('Aria: Wi-Fi keep-on set', flush=True)
        else:
            detail = (result.stdout + result.stderr).strip().splitlines()[-1:] if result else ['timed out']
            print('Aria: could not set Wi-Fi keep-on:', *detail, flush=True)

    def start(self, folder):
        """Record into the session folder once `recording` is set."""
        self.folder = Path(folder)

    def device_minus_host_ns(self):
        """Aria device clock minus host monotonic clock (ns), or None before time sync is up."""
        sync = self.receiver.get_time_sync() if self.receiver else None
        if sync is None or not sync.is_valid():
            return None
        now = time.monotonic_ns()
        return sync.remote_timestamp_ns_from_local(now) - now

    def info(self):
        return {'serial': self.device.serial() if self.device else None,
                'profile': self.profile, 'interface': self.interface,
                'rgb_size': self.frame_size, 'video_fps_nominal': self.fps,
                'vrs_dir': 'aria_vrs' if self.vrs else None,
                'device_minus_host_ns': self.device_minus_host_ns()}

    def _on_rgb(self, image_data, record):
        arrival = time.monotonic()
        with self.sdk_lock:
            if self.stopping:
                return
            rgb = image_data.to_numpy_array()
            capture_ns = record.capture_timestamp_ns
            offset = self.device_minus_host_ns() if self.recording.is_set() else None
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.frame_size = [frame.shape[1], frame.shape[0]]
        if not self.recording.is_set():
            with self.lock:
                self.shared['aria_image'] = frame
            return
        start = self.shared['start']
        stamp = arrival - start
        capture_stamp = None if offset is None else (capture_ns - offset) / 1e9 - start
        with self.write_lock:
            if self.closed:
                return
            if self.writer is None:
                h, w = frame.shape[:2]
                self.writer = cv2.VideoWriter(str(self.folder / 'aria_rgb.mp4'),
                                              cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
                if not self.writer.isOpened():
                    raise RuntimeError('Cannot create aria_rgb.mp4')
                self.csv_file = (self.folder / 'aria_rgb_timestamps.csv').open('w', newline='')
                self.csv_out = csv.writer(self.csv_file)
                self.csv_out.writerow(['aria_frame', 'host_elapsed_s', 'host_utc_iso',
                                       'aria_capture_ns', 'capture_host_elapsed_s',
                                       'device_minus_host_ns'])
            self.writer.write(frame)
            self.csv_out.writerow([self.index, f'{stamp:.6f}', datetime.now(timezone.utc).isoformat(),
                                   capture_ns, '' if capture_stamp is None else f'{capture_stamp:.6f}',
                                   '' if offset is None else offset])
            if self.index % max(1, self.fps) == 0:
                self.csv_file.flush()
            index = self.index
            self.index += 1
        with self.lock:
            self.shared['aria_image'] = frame
            self.shared['aria_frame'] = index
            self.shared['aria_stamp'] = capture_stamp if capture_stamp is not None else stamp

    def close(self):
        with self.sdk_lock:
            self.stopping = True
        if self.streaming:
            try:
                self.device.stop_streaming()
            except Exception as exc:
                print('Aria stop_streaming warning:', exc)
            self.streaming = False
            time.sleep(0.5)  # let in-flight frames drain
        if self.receiver is not None:
            try:
                self.receiver.stop_server()
            except Exception as exc:  # e.g. start() failed before the server was up
                print('Aria stop_server warning:', exc)
            self.receiver = None
        with self.write_lock:
            self.closed = True
            if self.writer is not None:
                self.writer.release()
            if self.csv_file is not None:
                self.csv_file.close()
        if self.client is not None and self.device is not None:
            try:
                self.client.disconnect(self.device)
            except Exception:
                pass
        self._move_vrs()

    def _move_vrs(self):
        if self.folder is None or self.vrs_staging is None or not self.vrs_staging.is_dir():
            return
        dest = self.folder / 'aria_vrs'
        dest.mkdir(exist_ok=True)
        for f in sorted(self.vrs_staging.iterdir()):
            target = dest / f.name.replace(':', '-')
            try:
                shutil.move(str(f), str(target))
                print('Aria VRS:', target)
            except OSError as exc:
                print(f'Could not move {f} to {target} ({exc}); it is still in {self.vrs_staging}')
                return
        try:
            self.vrs_staging.rmdir()
        except OSError:
            pass
