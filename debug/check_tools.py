from vega_client import PolarisVega

vega = PolarisVega()

try:
    vega.connect()

    print("INIT:", vega.command("INIT"))

    for command in [
        "PHSR 00",
        "PHSR 01",
        "PHSR 02",
        "PHSR 03",
    ]:
        print(f"{command}: {vega.command(command)}")

finally:
    vega.close()
