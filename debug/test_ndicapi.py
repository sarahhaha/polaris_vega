
import ctypes
from pathlib import Path

# Find the compiled shared library
matches = list(
    (Path.home() / "ndicapi" / "build").rglob("libndicapi.so")
)

if not matches:
    raise RuntimeError("libndicapi.so not found. Check CMake build.")

lib = ctypes.CDLL(str(matches[0]))

# Declare function argument and return types
lib.ndiOpenNetwork.argtypes = [ctypes.c_char_p, ctypes.c_int]
lib.ndiOpenNetwork.restype = ctypes.c_void_p

lib.ndiCommand.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.ndiCommand.restype = ctypes.c_char_p

lib.ndiCloseNetwork.argtypes = [ctypes.c_void_p]
lib.ndiCloseNetwork.restype = None

# Connect
device = lib.ndiOpenNetwork(b"169.254.82.86", 8765)

if not device:
    raise RuntimeError("Cannot connect to Polaris Vega")

print("Connected to Polaris Vega")

def command(cmd):
    response = lib.ndiCommand(device, cmd.encode("ascii"))
    result = response.decode("ascii") if response else None
    print(f"{cmd} -> {result}")
    return result

try:
    command("APIREV:")
    command("INIT:")
    command("PHSR:00")
finally:
    lib.ndiCloseNetwork(device)
    print("Connection closed")
