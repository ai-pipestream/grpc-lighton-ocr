#!/usr/bin/env python3
"""Required real-model RPC qualification. Missing models are failures, never skips.
Usage: native_rpc_test.py SERVER MODEL_DIRECTORY [FIXTURE]
"""
import base64
import json
import os
import pathlib
import socket
import struct
import subprocess
import sys
import time

root = pathlib.Path(__file__).resolve().parent.parent
server, models = sys.argv[1:3]
fixture = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else root/'tests/fixtures/native-invoice.png'
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    address = f'127.0.0.1:{sock.getsockname()[1]}'
log = open('/tmp/lighton-native-rpc-server.log', 'w')
external = os.environ.get('LIGHTON_RPC_ADDRESS')
if external:
    address = external
process = None if external else subprocess.Popen([server], env=dict(os.environ, GRPC_LIGHTON_LISTEN_ADDRESS=address,
    GRPC_LIGHTON_MODELS_DIR=models, GRPC_LIGHTON_DIAGNOSTIC_NO_ENGINE='0'), stdout=log, stderr=log)
args = ['grpcurl', '-plaintext', '-import-path', str(root/'proto'), '-proto',
        'ai/pipestream/lightonocr/v1/lighton_ocr.proto']
service = 'ai.pipestream.lightonocr.v1.LightOnOcrService/'
def call(method, data, timeout=120):
    return subprocess.run(args + ['-max-time', str(timeout), '-d', '@', address, service+method],
                          input=data, text=True, capture_output=True, timeout=timeout+5)
def events(text):
    result = []
    while text.strip():
        obj, end = json.JSONDecoder().raw_decode(text.lstrip())
        result.append(obj); text = text.lstrip()[end:]
    return result
def payload(image, pages=1, **options):
    opts = dict(model='LIGHTON_MODEL_0_8B', pageCount=pages,
                emitPageDocuments=True, emitDocument=True, **options)
    return '\n'.join([json.dumps({'options':opts})] + [json.dumps({'page':{
        'pageNo': n, 'png':base64.b64encode(image).decode()}}) for n in range(1,pages+1)])
try:
    deadline = time.monotonic()+60
    while True:
        info = call('GetServiceInfo', '{}', 2)
        if info.returncode == 0:
            break
        if (process is not None and process.poll() is not None) or time.monotonic() > deadline:
            raise RuntimeError('startup failed: '+info.stderr)
        time.sleep(.1)
    info = json.loads(info.stdout)
    assert info['engineLoaded'] and info['provenance']['device'] == 'cpu', info
    assert len(info['provenance']['modelSha256']) == 64
    health = subprocess.run(['grpcurl', '-plaintext', '-import-path', str(root/'tests'),
        '-proto', 'health.proto', '-d', json.dumps({'service':service.rstrip('/')}),
        address, 'grpc.health.v1.Health/Check'], capture_output=True, text=True, check=True)
    assert json.loads(health.stdout)['status'] == 'SERVING'
    print('loaded-model service-qualified health PASS')
    image = fixture.read_bytes()
    response = call('ParseDocument', payload(image, pages=2))
    assert response.returncode == 0, response.stderr
    pathlib.Path('/tmp/lighton-native-rpc-response.jsonl').write_text(response.stdout)
    result = events(response.stdout)
    pages = [e['pageDocument'] for e in result if 'pageDocument' in e]
    assert len(pages) == 2, result
    for number, page in enumerate(pages, 1):
        assert page['pageNo'] == number
        assert page['provenance'] == info['provenance']
        document = page['document']
        text = ' '.join(next(iter(t.values()))['base']['text'] for t in document['texts'])
        assert all(word in text for word in ['INVOICE', '1042', '125.00', 'Paid']), text
        for item in document['texts']:
            base = next(iter(item.values()))['base']
            assert base['source'][0]['collector']['model'] == info['modelRepo']
            assert base['source'][0]['collector']['version'] == info['provenance']['modelRevision']
            for prov in base['prov']:
                assert prov['charspan'].get('start', 0) == 0
                assert prov['charspan']['end'] == len(base['text'])
                box = prov['bbox']
                assert prov['pageNo'] == number
                assert 0 <= box.get('l',0) < box['r'] <= 1000, box
                assert 0 <= box.get('t',0) < box['b'] <= 300, box
        assert document['generationTraces'][0]['steps']
    folded = next(e['document'] for e in result if 'document' in e)
    refs = [next(iter(t.values()))['base']['selfRef'] for t in folded['texts']]
    assert len(refs) == len(set(refs))
    assert [c['ref'] for c in folded['body']['children']] == refs
    assert result[-1]['complete']['pagesOk'] == 2
    print('real RPC: two pages, text, bounded boxes, provenance, traces and folded refs PASS')
    over = call('ParseDocument', payload(image, maxTokensPerPage=1025))
    assert over.returncode != 0 and 'ResourceExhausted' in over.stderr, over.stderr
    bomb = bytearray(image); bomb[16:24] = struct.pack('>II',10000,10000)
    over = call('ParseDocument', payload(bomb))
    assert over.returncode == 0 and 'PAGE_FAILURE_CODE_RESOURCE_LIMIT' in over.stdout, over.stderr+over.stdout
    short = call('ParseDocument', payload(image, maxTokensPerPage=1))
    assert short.returncode == 0 and 'PAGE_FAILURE_CODE_RESOURCE_LIMIT' in short.stdout, short.stderr+short.stdout
    partial = next(e['pageFailure'] for e in events(short.stdout) if 'pageFailure' in e)
    assert partial['finishReason'] == 'FINISH_REASON_TOKEN_LIMIT'
    assert partial['generationTrace']['steps'] and partial['generationTrace']['text']
    greedy = call('ParseDocument', payload(image, temperature=0))
    assert greedy.returncode == 0 and 'pageDocument' in greedy.stdout, greedy.stderr+greedy.stdout
    greedy_steps = next(e['pageDocument']['document']['generationTraces'][0]['steps']
                        for e in events(greedy.stdout) if 'pageDocument' in e)
    assert all(step['token'] == step['modelToken'] for step in greedy_steps)
    print('token-limited raw output retained; explicit-zero greedy sampling PASS')
    concurrent = subprocess.Popen(args + ['-max-time', '120', '-d', '@', address,
        service+'ParseDocument'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    concurrent.stdin.write(payload(image)); concurrent.stdin.close(); concurrent.stdin = None
    try:
        time.sleep(.3)
        other = call('ParseDocument', payload(image))
        assert other.returncode != 0 and 'ResourceExhausted' in other.stderr, other.stderr+other.stdout
        first, err = concurrent.communicate(timeout=120)
        assert concurrent.returncode == 0 and 'pageDocument' in first, err+first
    finally:
        if concurrent.poll() is None:
            concurrent.kill(); concurrent.wait()
    print('single-inference admission rejects concurrent decode PASS')
    cancelled = call('ParseDocument', payload(image), .2)
    assert cancelled.returncode != 0 and 'DeadlineExceeded' in cancelled.stderr, cancelled.stderr
    # Cancellation must release admission promptly, not poison the next context.
    time.sleep(1)
    after = call('ParseDocument', payload(image))
    assert after.returncode == 0 and 'pageDocument' in after.stdout, after.stderr+after.stdout
    print('pixel/token bounds and deadline cancellation with subsequent inference PASS')
finally:
    if process is not None:
        process.terminate()
        try:
            assert process.wait(timeout=7) == 0
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
    log.close()
