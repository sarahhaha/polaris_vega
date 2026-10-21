from vega_client import PolarisVega

vega = PolarisVega(host="169.254.51.30")

def send(cmd):
    response = vega.command(cmd)
    print(f"{cmd} -> {response}")
    if response.startswith("ERROR"):
        raise RuntimeError(f"NDI command failed: {response}")
    return response

tracking_started = False

try:
    vega.connect()

    # Initialize the tracking system
    send("INIT")

    # Query available tool handles
    send("PHSR 00")

    # Check which handles require initialization
    send("PHSR 02")

    # Check which handles require enabling
    send("PHSR 03")

    # Attempt to start tracking
    send("TSTART")
    tracking_started = True

    # Request 3D positions of passive stray markers
    for i in range(10):
        response = send("TX 1000")

finally:
    if tracking_started:
        try:
            send("TSTOP")
        except Exception as e:
            print("TSTOP failed:", e)

    vega.close()
