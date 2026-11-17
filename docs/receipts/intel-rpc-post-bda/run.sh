#!/usr/bin/env bash
set -eo pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -u
cd /work/main/grpc-services/worktrees/grpc-lighton-ocr-intel-sycl
exec 9>/work/main/grpc-services/worktrees/intel-sycl-repos/model-test.lock
flock -w 1 9
out=receipts/rpc-post-bda-20261108
mkdir "$out"
server=/work/main/grpc-services/worktrees/grpc-asr-intel-sycl/build-intel/intel-lighton-rpc
export PATH="$PWD/build-deps:$PATH"
export LD_LIBRARY_PATH="$PWD/build-llama-intel/bin:$LD_LIBRARY_PATH"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export GRPC_LIGHTON_BACKEND=sycl GRPC_LIGHTON_DEVICE=SYCL0
export GRPC_LIGHTON_MODELS_DIR=/work/models/lighton-intel-qualification
export GRPC_LIGHTON_LISTEN_ADDRESS=127.0.0.1:51074
sha256sum "$server" build-llama-intel/observed-ggml-backend.cpp build-llama-intel/bin/libggml-base.so > "$out/identities.sha256"
uptime > "$out/host-load.txt"
pid=
trap 'if [[ -n "$pid" ]]; then kill -TERM "$pid" 2>/dev/null || true; wait "$pid" || true; fi' EXIT
for mode in observed off; do
 export GRPC_LIGHTON_OBSERVE_PLACEMENT=1
 [[ "$mode" == observed ]] || export GRPC_LIGHTON_OBSERVE_PLACEMENT=0
 "$server" >"$out/$mode-server.stdout" 2>"$out/$mode-server.stderr" </dev/null &
 pid=$!
 for ((i=0;i<60;i++)); do
  if grpcurl -plaintext -max-time 2 -d '{}' 127.0.0.1:51074 ai.pipestream.lightonocr.v1.LightOnOcrService/GetServiceInfo >"$out/$mode-ready.json" 2>"$out/$mode-ready.stderr"; then break; fi
  kill -0 "$pid"; sleep 1
 done
 python3 - "$out" "$mode" <<'PY'
import sys,pathlib,json
sys.path.insert(0,'tests')
import hardware_rpc_test as c
import hardware_cancellation_test as cancel
out=pathlib.Path(sys.argv[1]); mode=sys.argv[2]
raw,seconds=c.rpc('127.0.0.1:51074','ParseDocument',cancel.payload(2),180)
(out/(mode+'-two-page.jsonl')).write_text(raw)
info,_=c.rpc('127.0.0.1:51074','GetServiceInfo','{}',30)
(out/(mode+'-info.json')).write_text(info)
p=json.loads(info)['provenance']
cancel.validate_metadata(p,'sycl',mode=='off')
c.validate_output(c.decode_events(raw),p)
for s in p['stages']:
 assert s['deviceId']=='SYCL0' and s['deviceName']=='Intel(R) Arc(TM) Pro B70 Graphics',s
(out/(mode+'-two-page.assertions.json')).write_text(json.dumps({'pass':True,'mode':mode,'seconds_diagnostic_only':seconds,'scope':'two-page text authored geometry refs completion and mode-appropriate metadata; no benchmark'},indent=2)+'\n')
print('PASS',mode,'two-page RPC',flush=True)
PY
 extra=()
 [[ "$mode" == observed ]] || extra=(--unobserved-regression)
 python3 tests/hardware_cancellation_test.py --address 127.0.0.1:51074 --backend sycl --output "$out/$mode-cancel" "${extra[@]}" </dev/null
 kill -TERM "$pid"; wait "$pid"; pid=
done
printf '0\n' > "$out/exitcode"
printf 'PASS observed/off two-page + three cancellation/recovery cycles each; GPU released\n'
