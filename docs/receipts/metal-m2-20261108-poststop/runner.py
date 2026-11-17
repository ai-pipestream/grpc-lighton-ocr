import hashlib,json,os,pathlib,platform,socket,subprocess,sys,time
root=pathlib.Path('/Users/krickert/work/astra-metal-20261108')
repo=root/'grpc-lighton-ocr'; out=root/'lighton-poststop-bda85'
sys.path.insert(0,str(repo/'tests'))
import hardware_rpc_test as checks
from hardware_cancellation_test import payload
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
(out/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with socket.socket() as s:s.bind(('127.0.0.1',50074))
env=os.environ.copy();env.update(GRPC_LIGHTON_BACKEND='metal',GRPC_LIGHTON_MODELS_DIR=str(root/'models/lighton'),GRPC_LIGHTON_LISTEN_ADDRESS='127.0.0.1:50074',GRPC_LIGHTON_OBSERVE_PLACEMENT='1',VECLIB_MAXIMUM_THREADS='2',OMP_NUM_THREADS='2')
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
  (out/'observed-info.json').write_text(info)
  provenance=json.loads(info)['provenance']
  checks.validate_placement(provenance,'metal');checks.validate_output(checks.decode_events(raw),provenance)
  print(f'PASS two-page text/anchor geometry/references/completion/both-stage placement: {seconds:.3f}s',flush=True)
  with (out/'cancellation.log').open('w') as clog:
   result=subprocess.run(['python3',str(repo/'tests/hardware_cancellation_test.py'),'--address','127.0.0.1:50074','--backend','metal','--output',str(out/'cancellation')],stdout=clog,stderr=subprocess.STDOUT,timeout=240)
  print((out/'cancellation.log').read_text(),flush=True)
  assert result.returncode==0,'cancellation suite failed'
  (out/'result.json').write_text(json.dumps({'source':sha,'two_page_seconds':seconds,'two_page_assertions':'PASS','cancel_recovery_cycles':3,'cancel_recovery':'PASS','comparison_repeated':False},indent=2)+'\n')
finally:
 if server and server.poll() is None:
  server.terminate()
  try:server.wait(timeout=10)
  except subprocess.TimeoutExpired:server.kill();server.wait()
 print('Owned Metal server stopped; no CPU server started.',flush=True)
