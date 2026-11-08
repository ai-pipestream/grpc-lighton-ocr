import hashlib,json,os,pathlib,platform,socket,subprocess,sys,time
root=pathlib.Path('/Users/krickert/work/astra-metal-20261108')
repo=root/'grpc-lighton-ocr'; out=root/'lighton-unobserved-bda85'
out.mkdir(exist_ok=False)
test_tools=root/'lighton-unobserved-tools-809f8a7'
sys.path[:0]=[str(test_tools),str(repo/'tests')]
import hardware_rpc_test as checks
from hardware_cancellation_test import payload, validate_metadata
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert sha=='bda85f4dd28ac6bad822e11d7673175c0eb504ed'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
files=subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
manifest={'tested_source_commit':sha,'machine':platform.machine(),'macos':platform.mac_ver()[0], 'chip':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'ram_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)), 'rosetta':subprocess.check_output(['sysctl','-in','sysctl.proc_translated'],text=True).strip(),'source_sha256':{f:digest(repo/f) for f in files if f=='CMakeLists.txt' or f.startswith(('src/','cmake/','proto/','tests/','scripts/','tools/'))},'binary_sha256':{f:digest(repo/'build-metal'/f) for f in ['grpc-lighton-ocr-server','inference_control_test','service_test','scheduler_stop_test']},'models_sha256':{f:digest(root/'models/lighton'/f) for f in ['model-bf16.gguf','vision-bf16.gguf']},'performance_comparison':'Not repeated; previous paired performance belongs to source 1bfc1df0cdee5f17e211be07001996155facfad5.'}
assert manifest['binary_sha256']['grpc-lighton-ocr-server']=='07a47d205577dce75c8452cf5bd85e11a563ac0c351edf4d65847e792c6248a8'
manifest['test_commit']='809f8a7'
manifest['test_sha256']={f:digest(test_tools/f) for f in ['hardware_cancellation_test.py','hardware_cancellation_checks_test.py']}
manifest['mode']='unobserved-regression; configured Metal, no observed offload claim'
manifest['observe_placement']='0'
(out/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with socket.socket() as s:s.bind(('127.0.0.1',50074))
env=os.environ.copy();env.update(GRPC_LIGHTON_BACKEND='metal',GRPC_LIGHTON_MODELS_DIR=str(root/'models/lighton'),GRPC_LIGHTON_LISTEN_ADDRESS='127.0.0.1:50074',GRPC_LIGHTON_OBSERVE_PLACEMENT='0',VECLIB_MAXIMUM_THREADS='2',OMP_NUM_THREADS='2')
server=None
try:
 with (out/'server.log').open('w') as log:
  server=subprocess.Popen(['nice','-n','10',str(repo/'build-metal/grpc-lighton-ocr-server')],env=env,stdout=log,stderr=subprocess.STDOUT)
  end=time.monotonic()+90
  while True:
   assert server.poll() is None,'server exited during startup'
   try:
    cold,_=checks.rpc('127.0.0.1:50074','GetServiceInfo','{}',2)
    if json.loads(cold).get('engineLoaded'):break
   except subprocess.CalledProcessError:pass
   assert time.monotonic()<end,'startup deadline'
   time.sleep(.2)
  (out/'cold-info.json').write_text(cold)
  raw,seconds=checks.rpc('127.0.0.1:50074','ParseDocument',payload(2),180)
  (out/'two-page.jsonl').write_text(raw)
  info,_=checks.rpc('127.0.0.1:50074','GetServiceInfo','{}',30)
  (out/'unobserved-info.json').write_text(info)
  provenance=json.loads(info)['provenance']
  validate_metadata(provenance,'metal',True);checks.validate_output(checks.decode_events(raw),provenance)
  print(f'PASS two-page text/anchor geometry/references/completion/unverified stages: {seconds:.3f}s',flush=True)
  with (out/'cancellation.log').open('w') as clog:
   result=subprocess.run(['python3',str(test_tools/'hardware_cancellation_test.py'),'--address','127.0.0.1:50074','--backend','metal','--unobserved-regression','--output',str(out/'cancellation')],stdout=clog,stderr=subprocess.STDOUT,timeout=240,env={**os.environ,'PYTHONPATH':str(repo/'tests')})
  print((out/'cancellation.log').read_text(),flush=True)
  assert result.returncode==0,'cancellation suite failed'
  assert server.poll() is None,'server did not survive recovery suite'
  (out/'result.json').write_text(json.dumps({'source':sha,'two_page_seconds':seconds,'two_page_assertions':'PASS','cancel_recovery_cycles':3,'cancel_recovery':'PASS','comparison_repeated':False,'mode':'unobserved-regression','observed_offload_claim':False,'single_server_pid':server.pid,'server_alive_after_cycles':server.poll() is None},indent=2)+'\n')
finally:
 if server and server.poll() is None:
  server.terminate()
  try:server.wait(timeout=10)
  except subprocess.TimeoutExpired:server.kill();server.wait()
 print('Owned Metal server stopped; no CPU server started.',flush=True)
