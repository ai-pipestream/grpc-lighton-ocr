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
import tempfile

server, grpc_source = sys.argv[1:]
with socket.socket() as listener:
    listener.bind(('127.0.0.1', 0))
    address = f'127.0.0.1:{listener.getsockname()[1]}'
env = dict(os.environ, GRPC_LIGHTON_DIAGNOSTIC_NO_ENGINE='1', GRPC_LIGHTON_LISTEN_ADDRESS=address,
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

# Production startup must fail clearly instead of becoming healthy with no OCR.
for model in ['0.8b', '4b']:
    result = subprocess.run([server], env=dict(env, GRPC_LIGHTON_DIAGNOSTIC_NO_ENGINE='0',
        GRPC_LIGHTON_MODEL=model), capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and 'listening on' not in result.stdout
    expected = 'missing configured model artifact' if model == '0.8b' else '4B is unavailable'
    assert expected in result.stderr, result.stderr
with tempfile.TemporaryDirectory(prefix='lighton-invalid-model-') as directory:
    pathlib.Path(directory, 'model-bf16.gguf').write_bytes(b'not a model')
    result = subprocess.run([server], env=dict(env, GRPC_LIGHTON_DIAGNOSTIC_NO_ENGINE='0',
        GRPC_LIGHTON_MODELS_DIR=directory, GRPC_LIGHTON_MODEL='0.8b'),
        capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and 'SHA256 mismatch' in result.stderr, result.stderr
print('missing weights, unqualified 4B and wrong artifact hash fail startup PASS')
