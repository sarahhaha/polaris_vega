import socket


class PolarisVega:
    def __init__(self, host="169.254.51.30", port=8765):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        self.sock = socket.create_connection(
            (self.host, self.port), timeout=5
        )
        self.sock.settimeout(5)
        print(f"Connected to Polaris Vega at {self.host}")

    def command(self, cmd):
        # ASCII commands terminate with carriage return.
        self.sock.sendall((cmd + "\r").encode("ascii"))

        response = bytearray()
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Camera disconnected")
            response.extend(chunk)
            if b"\r" in chunk:
                break

        return response.split(b"\r", 1)[0].decode(
            "ascii", errors="replace"
        )

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None


if __name__ == "__main__":
    vega = PolarisVega()

    try:
        vega.connect()

        for cmd in ["APIREV ", "INIT", "PHSR 00"]:
            response = vega.command(cmd)
            print(f"{cmd.strip()} -> {response}")

    finally:
        vega.close()
