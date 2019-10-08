import socket
import json
from typing import Optional


def ping_pong(host: str, port: int, request: Optional[dict] = None, receive_buffer_size: int = 102400) -> dict:
    s = socket.socket()
    s.connect((host, port))
    body = json.dumps(request).encode('utf-8') if request is not None else bytes()
    s.send(body + b'\n')

    response = json.loads(s.recv(receive_buffer_size, socket.MSG_WAITALL))
    s.close()
    return response
