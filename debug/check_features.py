from vega_client import PolarisVega

vega = PolarisVega()

try:
    vega.connect()

    for cmd in ["APIREV ", "VER 4", "VER 5"]:
        try:
            print(f"{cmd.strip()} -> {vega.command(cmd)}")
        except Exception as e:
            print(f"{cmd.strip()} -> ERROR: {e}")

finally:
    vega.close()
