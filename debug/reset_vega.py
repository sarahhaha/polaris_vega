
from vega_client import PolarisVega

vega = PolarisVega()

try:
    vega.connect()

    # Stop tracking if it is still active.
    print("TSTOP:", vega.command("TSTOP"))

    # Return to initialized configuration state.
    print("INIT:", vega.command("INIT"))

    # Check remaining tool handles.
    print("PHSR 00:", vega.command("PHSR 00"))

finally:
    vega.close()
