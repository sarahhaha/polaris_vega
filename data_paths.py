"""Where recordings are saved. Shared with chopsticks/process_sessions.py, which reads them.

First one that is set: --data-dir, the CHOPSTICKS_DATA environment variable, DEFAULT_DATA_DIR.
"""
import os
from datetime import datetime
from pathlib import Path

DEFAULT_DATA_DIR = Path('/media/nayoung/Extreme/chopsticks_data')


def data_root(arg=None):
    """The data folder. An explicit --data-dir is created if needed. The default (or $CHOPSTICKS_DATA)
    must already exist, so an unplugged drive is an error instead of a stray folder somewhere else."""
    if arg:
        root = Path(arg).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return root
    root = Path(os.environ.get('CHOPSTICKS_DATA') or DEFAULT_DATA_DIR).expanduser()
    if not root.is_dir():
        raise SystemExit(f'Data folder {root} does not exist. Is the external drive plugged in and mounted?\n'
                         'Or choose another folder with --data-dir or $CHOPSTICKS_DATA.')
    return root


def new_session_folder(root):
    folder = Path(root) / datetime.now().strftime('%Y%m%d_%H%M%S')
    folder.mkdir(exist_ok=False)
    return folder
