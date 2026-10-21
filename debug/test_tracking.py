
from vega_client import PolarisVega
import time

vega = PolarisVega()

try:
    vega.connect()

    print("Initializing...")
    print("INIT ->", vega.command("INIT"))

    print("Starting tracking...")
    response = vega.command("TSTART")
    print("TSTART ->", response)

    if not response.startswith("OKAY"):
        raise RuntimeError("Tracking did not start")

    try:
        for i in range(20):
            response = vega.command("TX 1000")
            print(f"Frame {i}: {response}")
            time.sleep(0.1)
    finally:
        print("TSTOP ->", vega.command("TSTOP"))

finally:
    vega.close()
