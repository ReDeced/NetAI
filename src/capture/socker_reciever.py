from __future__ import annotations
import socket
import struct
import os
import logging
import signal
from pathlib import Path

from src.network.network_event import NetworkEvent
from src.network.types import Endpoint, ProtocolType
from src.data.window_builder import WindowBuilder
from src.data.shard_writer import ShardWriter

log = logging.getLogger(__name__)

# Должен совпадать с #pragma pack(push,1) структурой в C++
# double(8) + 2*uint32(8) + 4*uint16(8) + uint8(1) = 25 байт
MSG_FORMAT = "=d II HHH B"
MSG_SIZE   = struct.calcsize(MSG_FORMAT)

PROTO_MAP = {
    0: ProtocolType.TCP,
    1: ProtocolType.UDP,
    2: ProtocolType.ICMP,
    3: ProtocolType.OTHER,
}


def parse_msg(data: bytes) -> NetworkEvent:
    ts, src_ip, dst_ip, src_port, dst_port, size, proto = \
        struct.unpack(MSG_FORMAT, data)

    def ip_str(packed: int) -> str:
        return socket.inet_ntoa(struct.pack("I", packed))

    return NetworkEvent(
        protocol=PROTO_MAP.get(proto, ProtocolType.OTHER),
        source=Endpoint(ip=ip_str(src_ip), port=src_port or None),
        destination=Endpoint(ip=ip_str(dst_ip), port=dst_port or None),
        size=size,
        timestamp=ts,
    )


def recv_exact(conn: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def run(
    socket_path: str = "/tmp/netai.sock",
    output_dir: Path = Path("data/raw"),
):
    # Удаляем старый сокет если остался
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    builder = WindowBuilder(timeout=60.0, max_len=128, min_len=32)
    writer  = ShardWriter(output_dir, shard_size=100_000)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)

    log.info(f"Listening on {socket_path} ...")

    def shutdown(*_):
        log.info("Shutting down...")
        writer.flush()
        server.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        raise SystemExit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    conn, _ = server.accept()
    log.info("C++ sniffer connected.")

    while True:
        raw = recv_exact(conn, MSG_SIZE)
        if raw is None:
            log.info("Sniffer disconnected.")
            break

        event = parse_msg(raw)
        result = builder.process(event)

        if result is not None:
            window, ts = result
            writer.add(window, ts)

            if builder.total_windows % 5000 == 0:
                builder.manager.cleanup()
                log.info(f"Windows: {builder.total_windows}")

    writer.flush()
    conn.close()
    server.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
