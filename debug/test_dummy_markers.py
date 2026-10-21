from vega_client import PolarisVega
import time

vega = PolarisVega(host="169.254.51.30")
tracking = False

def send(cmd):
    response = vega.command(cmd)
    print(f"{cmd} -> {repr(response)}")

    if response.startswith("ERROR"):
        raise RuntimeError(response)

    return response

try:
    vega.connect()

    # Initialize camera
    send("INIT")

    # Request passive dummy tool
    reply = send("PHRQ *********1**01")

    handle = reply[:2]
    print("Dummy tool handle:", handle)

    # Initialize and enable
    send(f"PINIT {handle}")
    send(f"PENA {handle}D")

    # Verify enabled handles
    send("PHSR 04")
    send("GET Info.Status.Alerts")
    # Start tracking
    send("TSTART")
    tracking = True

    # Compare tracking responses
    for cmd in ["TX 1000", "TX 1800"]:
        response = send(cmd)
        time.sleep(0.1)

finally:
    if tracking:
        try:
            send("TSTOP")
        except Exception as e:
            print("TSTOP error:", e)

    vega.close()
