from __future__ import annotations

import socket

for port in range(8501, 8521):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        raise SystemExit(0)
raise SystemExit("No free localhost port found in 8501-8520")
