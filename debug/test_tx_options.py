from vega_client import PolarisVega

vega = PolarisVega()

try:
    vega.connect()

    print("INIT:", vega.command("INIT"))
    print("TSTART:", vega.command("TSTART"))

    try:
        for cmd in [
            "TX 0001",
            "TX 0800",
            "TX 1000",
            "TX 1800",
        ]:
            print(f"{cmd} -> {vega.command(cmd)}")
    finally:
        print("TSTOP:", vega.command("TSTOP"))

finally:
    vega.close()
