# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0

"""Allocate independent service, rendezvous and HCCL port blocks for a test."""

import errno
import socket
from contextlib import ExitStack


def _reserve_ports(stack: ExitStack, count: int) -> int:
    for _ in range(32):
        attempt = ExitStack()
        try:
            first = attempt.enter_context(socket.socket())
            first.bind(("0.0.0.0", 0))
            base = first.getsockname()[1]
            if base + count - 1 > 65535:
                attempt.close()
                continue
            for offset in range(1, count):
                sock = attempt.enter_context(socket.socket())
                sock.bind(("0.0.0.0", base + offset))
            stack.enter_context(attempt.pop_all())
            return base
        except OSError as error:
            attempt.close()
            if error.errno != errno.EADDRINUSE:
                raise
    raise RuntimeError("No available test port block after 32 attempts")


if __name__ == "__main__":
    with ExitStack() as ports:
        # Release only after choosing mutually disjoint listener ranges.
        print(*(_reserve_ports(ports, size) for size in (1, 32, 64)))
