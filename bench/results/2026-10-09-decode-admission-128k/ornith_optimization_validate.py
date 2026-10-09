import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path.cwd()
data=root.parent/'Lamina-data'
out=data/'performance/ornith-optimization/validation'
out.mkdir(parents=True,exist_ok=True)
model=str(data/'models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')
env=dict(os.environ)
for key in ('LAMINA_PROFILE','LAMINA_TIMELINE','LAMINA_MOE_GRAPH_CACHE','LAMINA_ADMIT_MB','LAMINA_CPU_THREADS','LAMINA_PREFILL_PROGRESS'):
    env.pop(key,None)
env['OPENBLAS_NUM_THREADS']='1'
checks=[
 ('reference-40',[sys.executable,'-m','tools.reference_prefix','--layers','40','--engine','build-cuda/lamina-infer.exe','--cuda','--batched','--tokens','42','43','44','45','--max-context','131072','--kv-cache','host'],{}),
 ('elementwise',['build-cuda/lamina-cuda-elementwise-check.exe'],{}),
 ('attention',['build-cuda/lamina-cuda-attention-check.exe'],{}),
 ('projection',['build-cuda/lamina-cuda-projection-check.exe',model],{}),
 ('prefix-default',['build-cuda/lamina-prefix-check.exe',model],{}),
 ('prefix-eviction',['build-cuda/lamina-prefix-check.exe',model],{'LAMINA_MOE_GRAPH_CACHE':'1'}),
 ('prefill',['build-cuda/lamina-prefill-check.exe',model],{}),
 ('cpu-configure',['cmake','-S','.','-B','build'],{}),
 ('cpu-build',['cmake','--build','build','--config','Release','--target','lamina-gguf','lamina-infer','lamina-sampling-check'],{}),
 ('python-tests',[sys.executable,'-m','unittest','discover','-s','tests/lamina'],{}),
 ('sampling',['build/Release/lamina-sampling-check.exe'],{}),
]
results=[]
for name,command,extra in checks:
    print('START',name,flush=True)
    started=time.perf_counter()
    with (out/(name+'.txt')).open('w',encoding='utf-8') as log:
        process=subprocess.run(command,cwd=root,env={**env,**extra},stdout=log,stderr=subprocess.STDOUT)
    result={'check':name,'command':command,'environment_overrides':extra,'returncode':process.returncode,'seconds':time.perf_counter()-started}
    results.append(result)
    (out/'results.json').write_text(json.dumps(results,indent=2))
    print('DONE',name,process.returncode,round(result['seconds'],2),flush=True)
    if process.returncode:
        print((out/(name+'.txt')).read_text(encoding='utf-8')[-5000:],flush=True)
        raise SystemExit(process.returncode)
