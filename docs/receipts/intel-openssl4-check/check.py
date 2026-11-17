import pathlib,subprocess,json,shlex,hashlib
root=pathlib.Path('/work/main/grpc-services/worktrees/grpc-asr-intel-sycl/build-intel')
src=root/'_deps/grpc-src'; dest=pathlib.Path('/tmp/lighton-openssl-check')
names=['src/core/tsi/ssl_transport_security.cc','src/core/tsi/ssl_transport_security_utils.cc','src/core/credentials/transport/tls/grpc_tls_crl_provider.cc']
for name in names:
 p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(subprocess.check_output(['git','-C',str(src),'show','HEAD:'+name]))
entries=json.loads((root/'compile_commands.json').read_text())
def check(label):
 results=[]
 for name in names:
  target='_deps/grpc-build/CMakeFiles/grpc.dir/'+name+'.o'
  command=subprocess.check_output(['ninja','-C',str(root),'-t','commands',target],text=True).splitlines()[-1]
  e={'file':str(src/name),'directory':str(root)};args=shlex.split(command);out=[];skip=False
  for a in args:
   if skip:skip=False;continue
   if a in ['-o','-MF','-MT']:skip=True;continue
   if a in ['-c','-MD']:continue
   out.append(str(dest/name) if a==e['file'] else a)
  out+=['-fsyntax-only'];r=subprocess.run(['nice','-n','10']+out,cwd=e['directory'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  (dest/(label+'-'+pathlib.Path(name).name+'.log')).write_bytes(r.stdout);results.append(r.returncode)
 print(label,results,flush=True);return results
red=check('red')
script=dest/'patch.cmake';script.write_text('set(grpc_SOURCE_DIR "'+str(dest)+'")\ninclude(/tmp/lighton-PatchGrpcOpenSSL.cmake)\n')
subprocess.run(['cmake','-P',str(script)],check=True)
green=check('green')
assert all(x!=0 for x in red) and green==[0,0,0]
(dest/'sources.sha256').write_text(''.join(hashlib.sha256((dest/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in names))
