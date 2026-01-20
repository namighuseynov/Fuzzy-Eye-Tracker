import socket
import json


class UdpSender:
    def __init__(self, host = "127.0.0.1", port = 505):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload: dict):
        msg = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.sock.sendto(msg, self.addr)

    def close(self):
        try:
            self.sock.close()
        except:
            pass