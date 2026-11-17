#!/usr/bin/env python3
"""Offline-only model preparation. Not imported or executed by the service.
Use a Python environment with CPU torch 2.11.0 and transformers 5.5.4.
Usage: prepare_model.py PINNED_LLAMA_CHECKOUT OUTPUT_DIRECTORY [PRECISION]
PRECISION is bf16 (default, the CPU backend) or f16 (OpenVINO devices: the
OpenVINO GPU plugin does not take BF16 weights). Both convert the same pinned
source files and are checked against models/SHA256SUMS.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import urllib.request

root = pathlib.Path(__file__).resolve().parent.parent
if len(sys.argv) not in (3, 4):
    raise SystemExit(__doc__)
llama, output = map(pathlib.Path, sys.argv[1:3])
precision = sys.argv[3] if len(sys.argv) == 4 else 'bf16'
if precision not in ('bf16', 'f16'):
    raise SystemExit('PRECISION must be bf16 or f16')
revision = subprocess.check_output(['git','-C',str(llama),'rev-parse','HEAD'],text=True).strip()
if revision != 'f3586e66bc916d672773d4a6505513ba91a3595c':
    raise SystemExit('converter checkout must match the pinned native backend revision')
manifest = json.loads((root/'models/source-manifest.json').read_text())
source = output/'source'; source.mkdir(parents=True, exist_ok=True)
def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
for artifact in manifest['files']:
    path = source/artifact['file']
    if path.exists() and digest(path) == artifact['sha256']:
        continue
    temp = path.with_suffix(path.suffix+'.partial')
    with urllib.request.urlopen(artifact['url'], timeout=120) as response, temp.open('wb') as out:
        while chunk := response.read(1024*1024):
            out.write(chunk)
    if temp.stat().st_size != artifact['size'] or digest(temp) != artifact['sha256']:
        raise SystemExit('source hash or size mismatch: '+artifact['file'])
    temp.replace(path)
names = {f'model-{precision}.gguf': ['--no-mtp'], f'vision-{precision}.gguf': ['--mmproj']}
for name, flags in names.items():
    subprocess.run([sys.executable,str(llama/'convert_hf_to_gguf.py'),str(source),
        '--outtype',precision,'--outfile',str(output/name),*flags], check=True,
        env=dict(os.environ, OMP_NUM_THREADS='2'))
pinned = dict(reversed(line.split()) for line in (root/'models/SHA256SUMS').read_text().splitlines())
for name in names:
    if digest(output/name) != pinned[name]:
        raise SystemExit('converted hash mismatch: '+name)
print(f'verified source revision and converted {precision} model/projector hashes')
