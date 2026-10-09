import json
import os
from pathlib import Path
import subprocess
import sys

root=Path.cwd()
data=root.parent/'Lamina-data'
out=data/'performance/ornith-optimization/early-admit-screen'
env=dict(os.environ)
for k in ('LAMINA_PROFILE','LAMINA_TIMELINE','LAMINA_MOE_GRAPH_CACHE','LAMINA_CPU_THREADS','LAMINA_ADMIT_MB','LAMINA_PREFILL_PROGRESS'):
    env.pop(k,None)
for mtp in (False,True):
    for early in ('0','1'):
        name=('on' if mtp else 'off')+'-early'+early
        directory=out/name
        command=[sys.executable,'-m','tools.compare_lamina','--engine','build-cuda/lamina-infer.exe','--model',str(data/'models/Ornith-1.5-35B-Q4_K_M.gguf'),'--tokenizer',str(data/'ornith/tokenizer.json'),'--context','131072','--tokens','128','--repeats','1','--report-dir',str(directory)]
        if mtp:command+=['--mtp',str(data/'mtp/ornith-mtp.gguf')]
        print('START',name,flush=True)
        subprocess.run(command,env={**env,'LAMINA_ADMIT_EARLY':early},cwd=root,check=True)
        print('DONE',name,flush=True)
    before=out/('on' if mtp else 'off')/'unused'
    a=out/(('on' if mtp else 'off')+'-early0')
    b=out/(('on' if mtp else 'off')+'-early1')
    sa=json.loads((a/'summary.json').read_text());sb=json.loads((b/'summary.json').read_text())
    same=all(json.loads(f.read_text())['next_tokens']==json.loads((b/f.name).read_text())['next_tokens'] for f in a.glob('*-1.json'))
    print('RESULT',mtp,'gain_percent',100*(sb['median_later_tokens_per_second']/sa['median_later_tokens_per_second']-1),'tokens_identical',same,flush=True)
    assert same
