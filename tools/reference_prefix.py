"""Check a Lamina layer prefix against NumPy and gguf-py's dequantizers.

Uses tokens 42 and 43 to exercise recurrent state and attention KV caches.
Requires: python -m pip install -r requirements-reference.txt. The model may
be a cropped GGUF containing all global tensors and the selected layer prefix.
"""
import argparse
import functools
from pathlib import Path
import re
import subprocess
import numpy as np
import gguf
from tools.gguf_reader import GGUFFile
from tools.lamina_model import FILENAME

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', type=Path, default=root.parent / 'Lamina-data' / 'models' / FILENAME)
parser.add_argument('--engine', type=Path, default=root / 'build' / 'lamina-infer')
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()
if not 1 <= args.layers <= 40:
    parser.error('--layers must be 1..40')
path = args.model
gg = GGUFFile(path)
by_name = {t.name: t for t in gg.tensors}
mm = np.memmap(path, dtype=np.uint8, mode='r')

@functools.lru_cache(maxsize=24)
def values(name, expert=None):
    t = by_name[name]
    elements = int(np.prod(t.shape[:2])) if expert is not None else t.elements
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[t.type_name]]
    start = gg.data_start + t.offset + (expert or 0) * elements // block * size
    data = mm[start:start+elements//block*size]
    if t.type_name == 'F32': return np.frombuffer(data, dtype='<f4').copy()
    return gguf.dequantize(data, gguf.GGMLQuantizationType[t.type_name])

def mat(name, x, expert=None):
    t = by_name[name]
    return values(name, expert).reshape(t.shape[1], t.shape[0]) @ x

def rms(x,w):
    return x/np.sqrt(np.mean(x.astype(np.float64)**2,axis=-1,keepdims=True)+1e-6)*w

def l2(x):
    return x/np.sqrt((x*x).sum(axis=-1,keepdims=True)+1e-6)

def sig(x): return 1/(1+np.exp(-x))
def silu(x): return x*sig(x)
def softplus(x): return np.maximum(x,0)+np.log1p(np.exp(-np.abs(x)))

def rope(x,pos):
    freq=pos/10000000.0**(2*np.arange(32)/64)
    co=np.cos(freq); si=np.sin(freq)
    a=x[...,:32].copy(); b=x[...,32:64].copy()
    x[...,:32]=a*co-b*si; x[...,32:64]=a*si+b*co
    return x

conv={l:np.zeros((8192,3),dtype=np.float32) for l in range(args.layers) if l%4 != 3}
mem={l:np.zeros((32,128,128),dtype=np.float32) for l in range(args.layers) if l%4 != 3}
keys={l:[] for l in range(args.layers) if l%4 == 3}
vals={l:[] for l in range(args.layers) if l%4 == 3}
for pos,token in enumerate([42,43]):
    t=by_name['token_embd.weight']; block,size=gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[t.type_name]]
    start=gg.data_start+t.offset+token*(2048//block)*size
    x=gguf.dequantize(mm[start:start+(2048//block)*size],gguf.GGMLQuantizationType[t.type_name])
    for layer in range(args.layers):
        prefix=f'blk.{layer}.'
        n=rms(x,values(prefix+'attn_norm.weight'))
        if layer%4 != 3:
            qkv=mat(prefix+'attn_qkv.weight',n)
            z=mat(prefix+'attn_gate.weight',n)
            alpha=mat(prefix+'ssm_alpha.weight',n)
            beta=sig(mat(prefix+'ssm_beta.weight',n))
            a=values(prefix+'ssm_a'); dt=values(prefix+'ssm_dt.bias')
            kernel=values(prefix+'ssm_conv1d.weight').reshape(8192,4)
            c=silu(np.sum(conv[layer]*kernel[:,:3],axis=1)+qkv*kernel[:,3])
            conv[layer][:,:2]=conv[layer][:,1:]; conv[layer][:,2]=qkv
            q=l2(c[:2048].reshape(16,128)); k=l2(c[2048:4096].reshape(16,128))
            v=c[4096:].reshape(32,128)
            kh=np.arange(32)%16
            decay=np.exp(a*softplus(alpha+dt))
            m=mem[layer]*decay[:,None,None]
            pred=np.einsum('hij,hi->hj',m,k[kh])
            delta=(v-pred)*beta[:,None]
            m+=k[kh,:,None]*delta[:,None,:]
            mem[layer]=m
            out=np.einsum('hij,hi->hj',m,q[kh])/np.sqrt(128)
            out=rms(out,values(prefix+'ssm_norm.weight'))*silu(z.reshape(32,128))
            mix=mat(prefix+'ssm_out.weight',out.reshape(-1))
        else:
            qfull=mat(prefix+'attn_q.weight',n).reshape(16,2,256)
            q=rms(qfull[:,0,:].copy(),values(prefix+'attn_q_norm.weight'))
            gate=qfull[:,1,:]
            k=rms(mat(prefix+'attn_k.weight',n).reshape(2,256),values(prefix+'attn_k_norm.weight'))
            v=mat(prefix+'attn_v.weight',n).reshape(2,256)
            keys[layer].append(rope(k,pos)); vals[layer].append(v)
            q=rope(q,pos)
            attn=np.zeros((16,256),dtype=np.float32)
            for h in range(16):
                kh=h//8
                scores=np.array([np.dot(q[h],key[kh])/16 for key in keys[layer]])
                probs=np.exp(scores-scores.max()); probs/=probs.sum()
                for t,p in enumerate(probs): attn[h]+=p*vals[layer][t][kh]
            attn*=sig(gate)
            mix=mat(prefix+'attn_output.weight',attn.reshape(-1))
        x=x+mix
        n=rms(x,values(prefix+'post_attention_norm.weight'))
        router=mat(prefix+'ffn_gate_inp.weight',n)
        order=np.argsort(-router)[:8]
        weights=np.exp(router[order]-router[order].max()); weights/=weights.sum()
        moe=np.zeros(2048,dtype=np.float32)
        for i,e in enumerate(order):
            gate=mat(prefix+'ffn_gate_exps.weight',n,int(e))
            up=mat(prefix+'ffn_up_exps.weight',n,int(e))
            moe+=weights[i]*mat(prefix+'ffn_down_exps.weight',silu(gate)*up,int(e))
        shared_gate=sig(np.dot(values(prefix+'ffn_gate_inp_shexp.weight'),n))
        gate=mat(prefix+'ffn_gate_shexp.weight',n)
        up=mat(prefix+'ffn_up_shexp.weight',n)
        moe+=shared_gate*mat(prefix+'ffn_down_shexp.weight',silu(gate)*up)
        x=x+moe
        if args.verbose:
            print('token',token,'layer',layer,'experts',order.tolist())
result = subprocess.run([str(args.engine.resolve()), str(path.resolve()), '--prefix', str(args.layers), '42', '43'],
                        check=True, capture_output=True, text=True)
cpp=np.fromstring(result.stdout,sep=' ')
if len(cpp) != 2048:
    raise RuntimeError(f'native engine returned {len(cpp)} hidden values, expected 2048')
print('max_abs_diff',float(np.max(np.abs(x-cpp))))
print('mean_abs_diff',float(np.mean(np.abs(x-cpp))))
print('reference',x[:8]); print('native',cpp[:8])
if np.max(np.abs(x-cpp)) >= 1e-5:
    raise RuntimeError('native prefix differs from independent NumPy calculation')
if args.layers == 40:
    normalized = rms(x, values('output_norm.weight'))
    logits = mat('output.weight', normalized)
    reference_next = int(np.argmax(logits))
    run = subprocess.run([str(args.engine.resolve()), str(path.resolve()), '42', '43'],
                         check=True, capture_output=True, text=True)
    match = re.search(r'next=(\d+) logit=([-\d.]+)', run.stdout.splitlines()[-1])
    if not match:
        raise RuntimeError('native engine did not report final logits')
    native_next, native_logit = int(match.group(1)), float(match.group(2))
    print('reference_next', reference_next, 'reference_logit', float(logits[reference_next]))
    print('native_next', native_next, 'native_logit', native_logit)
    if native_next != reference_next or abs(native_logit - logits[reference_next]) >= 1e-3:
        raise RuntimeError('native final logits differ from independent NumPy calculation')
