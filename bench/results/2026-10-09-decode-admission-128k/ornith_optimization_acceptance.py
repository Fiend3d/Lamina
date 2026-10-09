import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root))
data = root.parent / 'Lamina-data'
out = data / 'performance/ornith-optimization/acceptance'
models = {
    'ornith': (data / 'models/Ornith-1.5-35B-Q4_K_M.gguf', data / 'ornith/tokenizer.json', data / 'mtp/ornith-mtp.gguf'),
    'qwen': (data / 'models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf', data / 'tokenizer/tokenizer.json', data / 'mtp/qwen36-mtp-q8_0.gguf'),
}
engines = {'before': root / 'build-cuda/lamina-infer-ornith-baseline.exe', 'after': root / 'build-cuda/lamina-infer.exe'}
env = dict(os.environ)
for k in ('LAMINA_PROFILE', 'LAMINA_DEV_PROFILE', 'LAMINA_TIMELINE', 'LAMINA_PAIR_PROFILE', 'LAMINA_CPU_THREADS', 'LAMINA_ADMIT_MB', 'LAMINA_ADMIT_PER_LAYER', 'LAMINA_MOE_GRAPH_CACHE'):
    env.pop(k, None)
env['LAMINA_HOST_REGISTER'] = '1'

def run(name, model, variant, mtp=False, repeats=3, extra=()):
    weights, tokenizer, head = models[model]
    directory = out / name
    cmd = [sys.executable, '-m', 'tools.compare_lamina', '--engine', str(engines[variant]), '--model', str(weights), '--tokenizer', str(tokenizer), '--context', '131072', '--repeats', str(repeats), '--tokens', '256', '--host-register', '1', '--report-dir', str(directory), *extra]
    if mtp:
        cmd += ['--mtp', str(head)]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'invocation.json').write_text(json.dumps({'command': cmd, 'environment': {k:v for k,v in env.items() if k.startswith('LAMINA_')}}, indent=2))
    print('START', name, flush=True)
    subprocess.run(cmd, check=True, cwd=root, env=env)
    print('DONE', name, flush=True)
    return json.loads((directory / 'summary.json').read_text())

def compare(model, mode):
    before = out / f'{model}-before-{mode}'
    after = out / f'{model}-after-{mode}'
    a = json.loads((before / 'summary.json').read_text())
    b = json.loads((after / 'summary.json').read_text())
    equal = all(json.loads(f.read_text())['next_tokens'] == json.loads((after / f.name).read_text())['next_tokens'] for f in before.glob('*-[0-9]*.json'))
    result = {'before': a['median_later_tokens_per_second'], 'after': b['median_later_tokens_per_second'], 'gain_percent': (b['median_later_tokens_per_second']/a['median_later_tokens_per_second']-1)*100, 'tokens_identical': equal}
    (out / f'{model}-{mode}-comparison.json').write_text(json.dumps(result, indent=2))
    print('COMPARISON', model, mode, result, flush=True)
    assert equal, 'Output changed across native builds'
    return result

def fixture(model):
    from tokenizers import Tokenizer
    from lamina import chat_prompt
    tokenizer = Tokenizer.from_file(str(models[model][1]))
    prefix = 'The unique vault code for station Amber is LAMINA847263. Keep this code for the final question.\n'
    question = '\nFirst state the unique vault code for station Amber. Then write a detailed, continuous travel journal about a scientist visiting a remote mountain village. Include at least ten substantial paragraphs about the journey, weather, villagers, observations and unexpected events. Do not conclude early.\n'
    records = [f'Field record {day}: the northern plot received {day*17%83} millimetres of rain. The soil temperature was {day*13%29+3} degrees. Researchers counted {day*19%997+20} seedlings in row {day%31+1}, inspected the measuring instruments and recorded the observations before leaving the station.\n' for day in range(1, 5000)]
    target = 131072-256
    def count(text):
        return len(tokenizer.encode(chat_prompt([{'role':'user','content':text}]), add_special_tokens=False).ids)
    low, high = 0, len(records)
    while low < high:
        mid = (low+high+1)//2
        if count(prefix+''.join(records[:mid])+question) <= target:
            low = mid
        else:
            high = mid-1
    body = prefix+''.join(records[:low])
    padding = 0
    for _ in range(10):
        text = body+' observation'*padding+question
        n = count(text)
        if n == target:
            break
        padding += target-n
    assert count(text) == target
    path = data / f'performance/ornith-optimization/{model}-full-128k.txt'
    path.write_text(text, encoding='utf-8')
    print('FIXTURE', model, target, flush=True)
    return path

parser = argparse.ArgumentParser()
parser.add_argument('phase', choices=['short','mtp','final','prose','long','early'])
args = parser.parse_args()
if args.phase == 'short':
    for model in models:
        run(f'{model}-before-off', model, 'before')
        run(f'{model}-after-off', model, 'after')
        result = compare(model, 'off')
        if model == 'ornith' and result['gain_percent'] < 1:
            raise SystemExit('No clear Ornith improvement; base-model tests not started')
        run(f'{model}-before-on', model, 'before', mtp=True, repeats=1)
        run(f'{model}-after-on', model, 'after', mtp=True, repeats=1)
        compare(model, 'on')
elif args.phase == 'mtp':
    for model in models:
        run(f'{model}-before-on', model, 'before', mtp=True)
        run(f'{model}-after-on', model, 'after', mtp=True)
        compare(model, 'on')
elif args.phase == 'final':
    import shutil
    for model in models:
        for mode in ('off', 'on'):
            source = out / f'{model}-after-{mode}'
            shutil.copytree(source, out / f'{model}-unconditional512-{mode}')
            run(f'{model}-after-{mode}', model, 'after', mtp=mode=='on')
            compare(model, mode)
elif args.phase == 'early':
    import shutil
    for model in models:
        for mode in ('off','on'):
            source=out/f'{model}-after-{mode}'
            previous=out/f'{model}-graph512-{mode}'
            shutil.copytree(source,previous)
            run(f'{model}-after-{mode}',model,'after',mtp=mode=='on')
            current=out/f'{model}-after-{mode}'
            same=all(json.loads(f.read_text())['next_tokens']==json.loads((current/f.name).read_text())['next_tokens'] for f in previous.glob('*-[0-9]*.json'))
            old=json.loads((previous/'summary.json').read_text())['median_later_tokens_per_second']
            new=json.loads((current/'summary.json').read_text())['median_later_tokens_per_second']
            result={'before':old,'after':new,'gain_percent':100*(new/old-1),'tokens_identical':same}
            (out/f'{model}-{mode}-early-comparison.json').write_text(json.dumps(result,indent=2))
            print('EARLY COMPARISON',model,mode,result,flush=True)
            assert same
            compare(model,mode)
        if model=='ornith' and new<old:
            raise SystemExit('No MTP gain; base-model tests not started')
elif args.phase == 'prose':
    for model in models:
        for variant in engines:
            run(f'{model}-{variant}-prose', model, variant, mtp=True, repeats=5, extra=['--prompts', 'prose'])
        compare(model, 'prose')
else:
    env['LAMINA_PREFILL_PROGRESS'] = '1'
    for model in models:
        prompt = fixture(model)
        for mode in ('off', 'on'):
            name = f'{model}-after-full-128k-{mode}'
            run(name, model, 'after', mtp=mode=='on', repeats=1, extra=['--prompt-file', str(prompt), '--fixed-tokens', '--warmup-tokens', '64', '--progress-file', str(out / name / 'progress.json')])
            r = json.loads((out / name / 'retrieval-1.json').read_text())
            assert r['prompt_tokens'] == 130816 and r['generated_tokens'] == 256
            assert r['retrieval_passed'], 'Early-history retrieval failed'
            assert not r['eos_positions'], 'Generation continued beyond EOS'
            print('FULL 128K PASSED', model, mode, flush=True)
