
from vega_client import PolarisVega
import time

vega = PolarisVega()

try:
    vega.connect()

    print("INIT:", vega.command("INIT"))
    print("PHSR:", vega.command("PHSR 00"))

    response = vega.command("TSTART")
    print("TSTART:", response)

    if not response.startswith("OKAY"):
        raise RuntimeError("Unable to start tracking")

    try:
        print("Move reflective markers in front of the camera.")
        print("Press Ctrl+C to stop.")

        while True:
            response = vega.command("TX 1000")
            print("TX 1000:", response)
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass

    finally:
        print("TSTOP:", vega.command("TSTOP"))

finally:
    vega.close()
