import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

root = Path.cwd()
sys.path.insert(0, str(root))
from tools.lamina_chat import Engine
import pynvml

data = root.parent / 'Lamina-data'
out = data / 'performance/ornith-optimization/acceptance/ornith-pi-80k-after'
out.mkdir(parents=True, exist_ok=True)
for key in ('LAMINA_PROFILE', 'LAMINA_TIMELINE', 'LAMINA_CPU_THREADS', 'LAMINA_MOE_GRAPH_CACHE', 'LAMINA_ADMIT_MB'):
    os.environ.pop(key, None)
os.environ['LAMINA_HOST_REGISTER'] = '1'
os.environ['LAMINA_PREFILL_PROGRESS'] = '1'
engine = Engine(data/'models/Ornith-1.5-35B-Q4_K_M.gguf', data/'ornith/tokenizer.json', root/'build-cuda/lamina-infer.exe', cuda=True, max_context=131072, kv_cache='device', kv_type='f16', compute_mode='fast', mtp=data/'mtp/ornith-mtp.gguf')
engine.log_requests = True
records = (data/'performance/ornith-optimization/ornith-full-128k.txt').read_text(encoding='utf-8').splitlines()
question = '\nWrite a complete Python implementation of a file indexing service with SQLite, incremental updates, file hashes, a command line interface and tests. Start with code. Provide substantial implementation details.'
def message_count(text):
    rendered = engine.template.render(messages=[{'role':'user','content':text}], tools=[], add_generation_prompt=True, enable_thinking=False, add_vision_id=False)
    return len(engine.tokenizer.encode(rendered, add_special_tokens=False).ids)
low, high = 0, len(records)-2
while low < high:
    mid = (low+high+1)//2
    if message_count('\n'.join(records[:mid])+question) <= 80000: low=mid
    else: high=mid-1
body='\n'.join(records[:low])
padding=0
for _ in range(10):
    content=body+' observation'*padding+question
    count=message_count(content)
    if count==80000: break
    padding += 80000-count
assert count==80000
(data/'performance/ornith-optimization/ornith-pi-80k.txt').write_text(content, encoding='utf-8')
pynvml.nvmlInit()
gpu=pynvml.nvmlDeviceGetHandleByIndex(0)
peak=[0]
done=threading.Event()
def monitor():
    while not done.wait(.05): peak[0]=max(peak[0],pynvml.nvmlDeviceGetMemoryInfo(gpu).used)
watcher=threading.Thread(target=monitor,daemon=True)
watcher.start()
results=[]
messages=[{'role':'user','content':content}]
try:
    # Initialize kernels before the measured first turn, as in the full-128K runs.
    engine._start_native()
    engine.native.command('RESET',True)
    token=engine.native.command('42')
    engine.native.generate(63,token)
    engine.native.command('RESET',True)
    for turn in range(3):
        events=list(engine.events(messages,max_tokens=256,temperature=0,enable_thinking=False))
        final=events[-1]
        answer=''.join(e.get('delta',{}).get('content','') for e in events)
        timings=final['timings']; usage=final['usage']
        seconds=timings['total_seconds']-timings['first_token_seconds']
        record={'turn':turn+1,'usage':usage,'timings':timings,'later_tokens_per_second':(usage['completion_tokens']-1)/seconds,'peak_total_gpu_mib':peak[0]/2**20,'content':answer,'finish_reason':final['finish_reason']}
        results.append(record)
        print('TURN',turn+1,json.dumps({k:v for k,v in record.items() if k!='content'}),flush=True)
        (out/'turns.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        if turn: assert timings['cached_prompt_tokens']>=79990
        messages += [{'role':'assistant','content':answer},{'role':'user','content':'Continue the implementation. Add robust error handling, concurrency control and complete runnable tests. Include full code.'}]
    (out/'metadata.json').write_text(json.dumps({'engine_sha256':hashlib.sha256(engine.executable.read_bytes()).hexdigest(),'context_capacity':131072,'model':str(engine.model),'kv_type':'f16','compute_mode':'fast','mtp':str(engine.mtp),'gpu':pynvml.nvmlDeviceGetName(gpu),'driver':pynvml.nvmlSystemGetDriverVersion(),'environment':{k:v for k,v in os.environ.items() if k.startswith('LAMINA_')}},indent=2))
finally:
    if engine.native: (out/'native-stderr.txt').write_text('\n'.join(engine.native.errors),encoding='utf-8')
    engine.close()
    done.set(); watcher.join(); pynvml.nvmlShutdown()
