#!/usr/bin/env python3
"""Stop the dedicated qualification container while its real OCR RPC runs."""
import base64
import json
import pathlib
import subprocess
import sys
import time

address, container = sys.argv[1:]
root = pathlib.Path(__file__).resolve().parent.parent
request = '\n'.join([json.dumps({'options': {'model':'LIGHTON_MODEL_0_8B',
    'pageCount':1,'emitPageDocuments':True}}), json.dumps({'page':{'pageNo':1,
    'png':base64.b64encode((root/'tests/fixtures/native-invoice.png').read_bytes()).decode()}})])
client = subprocess.Popen(['grpcurl','-plaintext','-max-time','120',
    '-import-path',str(root/'proto'),'-proto','ai/pipestream/lightonocr/v1/lighton_ocr.proto',
    '-d','@',address,'ai.pipestream.lightonocr.v1.LightOnOcrService/ParseDocument'],
    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
try:
    client.stdin.write(request);client.stdin.close();client.stdin=None
    time.sleep(.3)
    assert client.poll() is None, 'RPC must be active before stopping the container'
    start=time.monotonic()
    subprocess.run(['docker','stop','-t','7',container],check=True,capture_output=True,timeout=9)
    elapsed=time.monotonic()-start
    output,error=client.communicate(timeout=3)
    status=subprocess.check_output(['docker','wait',container],text=True).strip()
    assert status == '0', status
    assert elapsed < 8, elapsed
    assert client.returncode != 0, 'expected active RPC cancellation on shutdown'
    print(f'active-inference shutdown PASS: exit {status}, {elapsed:.3f}s; {error.strip()}')
finally:
    if client.poll() is None:
        client.kill();client.wait()
