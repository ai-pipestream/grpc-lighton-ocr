#!/usr/bin/env python3
"""Process-level scaffold readiness/shutdown regression; requires grpcurl.
Usage: runtime_health_test.py SERVER GRPC_SOURCE_DIRECTORY
"""
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time

server, grpc_source = sys.argv[1:]
with socket.socket() as listener:
    listener.bind(('127.0.0.1', 0))
    address = f'127.0.0.1:{listener.getsockname()[1]}'
env = dict(os.environ, GRPC_LIGHTON_LISTEN_ADDRESS=address,
           GRPC_LIGHTON_MODELS_DIR='/nonexistent-lighton-test-models')
process = subprocess.Popen([server], env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
try:
    args = ['grpcurl', '-plaintext', '-max-time', '2', '-import-path', grpc_source,
            '-proto', 'src/proto/grpc/health/v1/health.proto']
    def health(name):
        result = subprocess.run(args + ['-d', json.dumps({'service': name}), address,
                                'grpc.health.v1.Health/Check'], capture_output=True,
                                text=True, timeout=3, check=True)
        return json.loads(result.stdout)['status']
    deadline = time.monotonic() + 10
    while True:
        try:
            status = health('ai.pipestream.lightonocr.v1.LightOnOcrService')
            break
        except subprocess.CalledProcessError:
            if process.poll() is not None or time.monotonic() > deadline:
                raise
            time.sleep(.05)
    assert status == 'NOT_SERVING', status
    assert health('') == 'SERVING'
    start = time.monotonic()
    for _ in range(10000):
        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            break
    assert process.wait(timeout=7) == 0
    print(f'runtime_health_test ok; repeated signals; shutdown {time.monotonic()-start:.3f}s')
finally:
    if process.poll() is None:
        process.kill()
        process.wait()
    print(process.stdout.read(), end='')
