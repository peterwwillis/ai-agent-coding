#!/usr/bin/env python3
from __future__ import annotations
_T='--no-mmap'
_S='--mmap'
_R='--n-gpu-layers'
_Q='--ubatch-size'
_P='--batch-size'
_O='--flash-attn'
_N='--cache-type-v'
_M='--cache-type-k'
_L='--ctx-size'
_K='nvidia'
_J='replace'
_I='Darwin'
_H='    '
_G='._-'
_F=False
_E='auto'
_D='models:'
_C=True
_B='utf-8'
_A=None
import argparse,os,platform,shlex,struct,sys
from dataclasses import dataclass
from pathlib import Path
from typing import List,Optional
DEFAULT_N_GPU_LAYERS='40'
LOG_NAME_MAX=64
BATCH_AUTO=_E
HELP_EPILOG='Batch/ubatch guidance\n\nHardware Type                            Recommended -b   Recommended -ub   Why?\nHigh-End GPU (e.g., RTX 4090)            4096              1024 - 2048        Fully utilizes many CUDA cores; 2048 can reduce prompt processing time by ~25%.\nMid-Range GPU (8GB-12GB VRAM)           2048              512                Prevents OOM (Out of Memory) while maintaining decent speeds.\nCPU / Mixed Inference                   2048              1024               Can provide up to a 3x speed gain for MoE models (like Mixtral).\nApple Silicon (M2/M3 Max)               4096              1024               Efficiently uses high unified memory bandwidth.\n\nOptimization Strategy\n  For Speed (Prompt Processing): Increase -ub. Larger values allow the GPU to process more prompt tokens in parallel, though the benefit often plateaus at 2048.\n  For Memory (VRAM Constraints): Decrease -b and -ub. High values allocate more memory for the logits/embeddings buffer. If you hit an OOM error, lower both values to 512 or 256.\n  Special Case (Embeddings/Reranking): You must set -b and -ub to the same value, or the server may fail.\n\nStart with -b 2048 -ub 512. If your GPU memory (VRAM) is less than 50% full during processing, try doubling -ub to 1024 and check if your "tokens per second" (TPS) for prompt processing increases.\n'
def default_llama_cache_dir():
	D='.cache';A='llama.cpp';B=os.environ.get('LLAMA_CACHE')
	if B:return Path(B).expanduser()
	C=platform.system()
	if C==_I:return Path.home()/'Library'/'Caches'/A
	if C in{'Linux','FreeBSD','OpenBSD','NetBSD','AIX'}:E=os.environ.get('XDG_CACHE_HOME',str(Path.home()/D));return Path(E)/A
	return Path.home()/D/A
def default_llama_swap_config_path():return Path.home()/'.config'/'llama-swap'/'config.yaml'
def read_gguf_chat_template(path):
	F={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
	def C(f):
		A=f.read(4)
		if len(A)!=4:raise EOFError
		return struct.unpack('<I',A)[0]
	def B(f):
		A=f.read(8)
		if len(A)!=8:raise EOFError
		return struct.unpack('<Q',A)[0]
	def G(f):
		A=B(f);C=f.read(A)
		if len(C)!=A:raise EOFError
		return C.decode(_B,errors=_J)
	def D(f,vtype):
		A=vtype
		if A in F:f.seek(F[A],os.SEEK_CUR);return
		if A==8:E=B(f);f.seek(E,os.SEEK_CUR);return
		if A==9:
			G=C(f);H=B(f)
			for I in range(H):D(f,G)
			return
		raise ValueError(f"Unsupported GGUF value type: {A}")
	try:
		if not path.is_file():return
		with path.open('rb')as A:
			if A.read(4)!=b'GGUF':return
			J=C(A);K=B(A);H=B(A)
			for L in range(H):
				I=G(A);E=C(A)
				if I=='tokenizer.chat_template':
					if E==8:return G(A)
					D(A,E);return
				D(A,E)
	except Exception:return
def read_gguf_block_count(path):
	F={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8};I={0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',10:'<Q',11:'<q'}
	def C(f):
		A=f.read(4)
		if len(A)!=4:raise EOFError
		return struct.unpack('<I',A)[0]
	def B(f):
		A=f.read(8)
		if len(A)!=8:raise EOFError
		return struct.unpack('<Q',A)[0]
	def J(f):
		A=B(f);C=f.read(A)
		if len(C)!=A:raise EOFError
		return C.decode(_B,errors=_J)
	def K(f,vtype):
		A=I.get(vtype)
		if not A:return
		B=struct.calcsize(A);C=f.read(B)
		if len(C)!=B:raise EOFError
		return int(struct.unpack(A,C)[0])
	def D(f,vtype):
		A=vtype
		if A in F:f.seek(F[A],os.SEEK_CUR);return
		if A==8:E=B(f);f.seek(E,os.SEEK_CUR);return
		if A==9:
			G=C(f);H=B(f)
			for I in range(H):D(f,G)
			return
		raise ValueError(f"Unsupported GGUF value type: {A}")
	try:
		if not path.is_file():return
		with path.open('rb')as A:
			if A.read(4)!=b'GGUF':return
			M=C(A);N=B(A);L=B(A)
			for O in range(L):
				G=J(A);E=C(A)
				if G.endswith('.block_count')or G in{'block_count','n_layer'}:
					H=K(A,E)
					if H is _A:D(A,E);return
					return H
				D(A,E)
	except Exception:return
def template_supports_thinking(path):
	A=read_gguf_chat_template(path)
	if not A:return _F
	return'enable_thinking'in A
def yaml_key(name):A=name;B=all(A.isalnum()or A in _G for A in A);return A if B else"'"+A.replace("'","''")+"'"
def normalize_yaml_key_text(key):
	A=key;A=A.strip()
	if len(A)>=2 and A[0]==A[-1]and A[0]in("'",'"'):
		B=A[1:-1]
		if A[0]=="'":return B.replace("''","'")
		return B.replace('\\"','"')
	return A
def sanitize_log_stem(name):
	A=''.join(A if A.isalnum()or A in _G else'_'for A in name);A=A.strip(_G)
	if not A:return'model'
	return A[:LOG_NAME_MAX]
def parse_n_gpu_layers(value):
	A=value.strip().lower()
	if A==_E:return A
	if A.isdigit():return str(int(A))
	raise argparse.ArgumentTypeError("n-gpu-layers must be an integer or 'auto'")
def parse_batch_setting(value):
	A=value.strip().lower()
	if A==BATCH_AUTO:return A
	if A.isdigit()and int(A)>0:return str(int(A))
	raise argparse.ArgumentTypeError("batch size must be a positive integer or 'auto'")
@dataclass(frozen=_C)
class HardwareProfile:name:str;batch:int;ubatch:int;vram_gb:Optional[float]=_A
def read_int_file(path):
	try:return int(path.read_text(encoding=_B).strip())
	except(OSError,ValueError):return
def read_nvidia_vram_bytes():
	B=Path('/proc/driver/nvidia/gpus')
	if not B.is_dir():return
	for F in B.glob('*/information'):
		try:
			for C in F.read_text(encoding=_B,errors='ignore').splitlines():
				if'Video Memory'in C:
					H,G=C.split(':',1);A=G.strip().split()
					if len(A)>=2 and A[0].isdigit():
						D=int(A[0]);E=A[1].lower()
						if E.startswith('mb'):return D*1024*1024
						if E.startswith('gb'):return D*1024*1024*1024
		except OSError:continue
def detect_linux_gpu(log):
	A=log;E=Path('/sys/class/drm')
	if not E.is_dir():A('No /sys/class/drm detected; skipping GPU vendor detection.');return _A,_A
	for D in E.glob('card*/device/vendor'):
		try:C=D.read_text(encoding=_B).strip().lower()
		except OSError:continue
		A(f"Detected DRM vendor {C} at {D.parent}");F=D.parent
		if C=='0x1002':
			B=read_int_file(F/'mem_info_vram_total')
			if B:A(f"AMD VRAM total: {B/1024**3:.1f} GB")
			else:A('AMD VRAM total not available from mem_info_vram_total.')
			return'amd',B
		if C=='0x10de':
			B=read_nvidia_vram_bytes()
			if B:A(f"NVIDIA VRAM total: {B/1024**3:.1f} GB")
			else:A('NVIDIA VRAM total not available from /proc/driver/nvidia/gpus.')
			return _K,B
		if C=='0x8086':A('Intel GPU detected; treating as mixed inference profile.');return'intel',_A
	return _A,_A
def detect_hardware_profile(log):
	H='cpu/mixed';G='mid-range-gpu';B=log;C=platform.system();D=platform.machine().lower();B(f"Platform: {C} ({D})")
	if C==_I and D in{'arm64','aarch64'}:B('Apple Silicon detected; using Apple Silicon batch profile.');return HardwareProfile('apple-silicon',4096,1024)
	if C=='Linux':
		E,F=detect_linux_gpu(B);A=F/1024**3 if F else _A
		if E in{'amd',_K}:
			if A is not _A and A>=20:B('High-end GPU detected; using 4096/1024 batch profile.');return HardwareProfile('high-end-gpu',4096,1024,A)
			if A is not _A and A>=8:B('Mid-range GPU detected; using 2048/512 batch profile.');return HardwareProfile(G,2048,512,A)
			if A is not _A and A<8:B('Low VRAM GPU detected; using 1024/256 batch profile.');return HardwareProfile('low-vram-gpu',1024,256,A)
			B('GPU detected without VRAM info; using mid-range GPU profile.');return HardwareProfile(G,2048,512,A)
		if E=='intel':B('Intel GPU detected; using CPU/mixed batch profile.');return HardwareProfile(H,2048,1024)
		B('No discrete GPU detected; using CPU/mixed batch profile.')
	return HardwareProfile(H,2048,1024)
def auto_batch_settings(profile,model_size_gb,log):
	E=profile;D=model_size_gb;C=log;B=E.batch;A=E.ubatch;C(f"Auto batch baseline from profile '{E.name}': -b {B} -ub {A}");C(f"Model size: {D:.1f} GB")
	if D>=20:B=min(B,2048);A=min(A,512);C('Model >= 20 GB; capping batch/ubatch to 2048/512.')
	elif D>=12:B=min(B,2048);A=min(A,512);C('Model >= 12 GB; capping batch/ubatch to 2048/512.')
	elif D<=4:
		if B>=4096:A=max(A,2048);C('Small model; increasing ubatch to at least 2048.')
		elif B>=2048:A=max(A,1024);C('Small model; increasing ubatch to at least 1024.')
	else:C('No size-based batch adjustments needed.')
	if B<=A:B=max(A+1,A*2);C(f"Adjusted batch to keep batch > ubatch: -b {B} -ub {A}")
	return B,A
def resolve_batch_settings(batch_arg,ubatch_arg,allow_equal,profile,model_path,log):
	G=model_path;F=profile;C=log;B=ubatch_arg;A=batch_arg;H=G.stat().st_size/1024**3;I=_A;J=_A
	if A==BATCH_AUTO or B==BATCH_AUTO:
		if F is _A:raise ValueError('auto batch sizing requires a detected hardware profile')
		I,J=auto_batch_settings(F,H,C)
	D=I if A==BATCH_AUTO else int(A);E=J if B==BATCH_AUTO else int(B)
	if A!=BATCH_AUTO:C(f"Using explicit batch size: -b {D}")
	if B!=BATCH_AUTO:C(f"Using explicit ubatch size: -ub {E}")
	if allow_equal:
		C('Allowing batch == ubatch for embeddings/reranking mode.')
		if D<E:raise ValueError('batch size must be greater than or equal to ubatch size')
	elif D<=E:raise ValueError('batch size must be greater than ubatch size')
	if A==BATCH_AUTO or B==BATCH_AUTO:C(f"Auto batch settings for '{G.name}': -b {D} -ub {E} (profile: {F.name}, model {H:.1f} GB)")
	return D,E
def parse_flash_attn(value):
	A=value.strip().lower()
	if A in{'on','off',_E}:return A
	raise argparse.ArgumentTypeError('flash-attn must be one of: on, off, auto')
def find_models_block(lines):
	A=lines;B=_A
	for(F,G)in enumerate(A):
		if G.strip().startswith(_D):B=F;break
	if B is _A:return _A,len(A)
	D=len(A)
	for C in range(B+1,len(A)):
		E=A[C].lstrip()
		if not E or E.startswith('#'):continue
		if not A[C].startswith(' '):D=C;break
	return B,D
def existing_model_keys(lines,start,end):
	C=set()
	for A in lines[start+1:end]:
		if not A.strip()or A.lstrip().startswith('#'):continue
		if A.startswith('  ')and not A.startswith(_H):
			B=A.strip()
			if B.endswith(':')and not B.startswith('-'):D=B[:-1].strip();C.add(normalize_yaml_key_text(D))
	return C
def parse_models_entries(lines,start,end):
	E=lines;F=[];A=start+1
	while A<end:
		C=E[A]
		if C.startswith('  ')and not C.startswith(_H)and C.strip()and not C.lstrip().startswith('#'):
			D=C.strip()
			if D.endswith(':')and not D.startswith('-'):
				G=normalize_yaml_key_text(D[:-1].strip());H=A;A+=1
				while A<end:
					B=E[A]
					if B.startswith('  ')and not B.startswith(_H)and B.strip()and not B.lstrip().startswith('#'):
						if B.strip().endswith(':')and not B.strip().startswith('-'):break
					A+=1
				I=A;F.append((G,H,I));continue
		A+=1
	return F
def read_header(template_path):
	B=template_path
	if not B or not B.is_file():return[_D]
	D=B.read_text(encoding=_B).splitlines();A=[]
	for C in D:
		A.append(C)
		if C.strip().startswith(_D):return A
	A.append(_D);return A
def build_cmd(llama_server,model_path,thinking,ctx_size,flash_attn,cache_type_k,cache_type_v,n_gpu_layers,mmap,batch_size,ubatch_size,log_file):
	E=ubatch_size;D=batch_size;C=n_gpu_layers;B=thinking;A=[shlex.quote(llama_server),'--offline','--log-file',shlex.quote(str(log_file)),'--log-colors','off','--log-prefix','--log-timestamps','-m',shlex.quote(str(model_path)),_L,str(ctx_size),_M,shlex.quote(cache_type_k),_N,shlex.quote(cache_type_v),_O,shlex.quote(flash_attn)]
	if D is not _A and E is not _A:A.extend([_P,str(D),_Q,str(E)])
	if C is not _A:A.extend([_R,str(C)])
	A.append(_S if mmap else _T)
	if B is not _A:F='true'if B else'false';A.extend(['--chat-template-kwargs',f"'{{\"enable_thinking\":{F}}}'"])
	A.extend(['--port','${PORT}']);return' '.join(A)
def main():
	n='ies';m='mmap';l='q8_0';W='\n';P='store_true';D=argparse.ArgumentParser(description='Generate a llama-swap config from llama.cpp cached GGUF models.',formatter_class=argparse.RawDescriptionHelpFormatter,epilog=HELP_EPILOG);D.add_argument('--llama-server',default=os.environ.get('LLAMA_SERVER','llama-server'),help='Path to llama-server binary (default: llama-server or $LLAMA_SERVER).');D.add_argument('--models-dir',default=os.environ.get('LLAMA_MODELS_DIR',str(default_llama_cache_dir())),help='Directory containing GGUF models (default: llama.cpp cache or $LLAMA_MODELS_DIR).');D.add_argument('--output',default=os.environ.get('LLAMA_SWAP_CONFIG',str(default_llama_swap_config_path())),help="Write output YAML to this path (use '-' for stdout).");D.add_argument('--template',default=_A,help='Optional header template file (default: llama-swap.yaml.example next to this script).');D.add_argument(_L,type=int,default=2048,help='Context size (default: 2048).');D.add_argument(_O,type=parse_flash_attn,default='on',help='Flash attention mode: on, off, auto (default: on).');D.add_argument(_M,default=l,help='KV cache K quantization type (default: q8_0).');D.add_argument(_N,default=l,help='KV cache V quantization type (default: q8_0).');D.add_argument(_R,type=parse_n_gpu_layers,default=_A,help='Number of layers to offload to GPU (default: not passed unless set).');D.add_argument('--gpu-layer-autodetect',action=P,help='Auto-detect GPU layers from model metadata.');D.add_argument(_S,dest=m,action=P,default=_C,help='Enable memory-mapped model loading (default).');D.add_argument(_T,dest=m,action='store_false',help='Disable memory-mapped model loading.');D.add_argument(_P,type=parse_batch_setting,default=_A,help='Batch size (default: not passed unless set).');D.add_argument(_Q,type=parse_batch_setting,default=_A,help='Ubatch size (default: not passed unless set).');D.add_argument('--batch-size-autodetect',action=P,help='Auto-detect batch/ubatch size based on hardware and model.');D.add_argument('--allow-equal-batch',action=P,help='Allow batch size to equal ubatch size (for embeddings/reranking).');D.add_argument('--verbose',action=P,help='Print progress to stderr.');D.add_argument('--prune-missing',action=P,help='Remove model entries not present in the llama.cpp cache.');A=D.parse_args();B=(lambda msg:print(msg,file=sys.stderr))if A.verbose else lambda _msg:_A
	if(A.batch_size==BATCH_AUTO or A.ubatch_size==BATCH_AUTO)and not A.batch_size_autodetect:print("ERROR: batch size 'auto' requires --batch-size-autodetect.",file=sys.stderr);return 2
	if(A.batch_size is _A)!=(A.ubatch_size is _A):print('ERROR: --batch-size and --ubatch-size must be provided together.',file=sys.stderr);return 2
	X=default_llama_cache_dir()
	try:X.mkdir(parents=_C,exist_ok=_C)
	except OSError as Y:print(f"ERROR: unable to create log directory: {X} ({Y})",file=sys.stderr);return 2
	L=_A;M=A.batch_size;N=A.ubatch_size
	if A.batch_size_autodetect:
		if M is _A and N is _A:M=BATCH_AUTO;N=BATCH_AUTO;B('Batch size autodetect enabled; using auto batch/ubatch.')
		else:B('Batch size autodetect enabled, but explicit batch/ubatch provided; skipping auto.')
	elif M is _A and N is _A:B('Batch/ubatch not provided; leaving llama-server defaults.')
	else:B('Using explicit batch/ubatch values; autodetect disabled.')
	Z=A.n_gpu_layers
	if A.gpu_layer_autodetect:
		if Z is _A:B('GPU layer autodetect enabled; using GGUF block_count.')
		else:B('GPU layer autodetect enabled, but explicit n-gpu-layers provided; skipping auto.')
	elif Z is _A:B('n-gpu-layers not provided; leaving llama-server defaults.')
	else:B('Using explicit n-gpu-layers; autodetect disabled.')
	Q=Path(A.models_dir).expanduser()
	if not Q.is_dir():print(f"ERROR: models directory not found: {Q}",file=sys.stderr);return 2
	B(f"Scanning for .gguf models under: {Q}");a=sorted(A for A in Q.rglob('*.gguf')if A.is_file())
	if not a:print(f"ERROR: no .gguf files found under: {Q}",file=sys.stderr);return 3
	B(f"Found {len(a)} .gguf model(s).");b=Path(A.template).expanduser()if A.template else Path(__file__).with_name('llama-swap.yaml.example');o=read_header(b)
	if b.is_file():B(f"Using template header from: {b}")
	else:B('Template header not found; using minimal header.')
	g={};O=[]
	for J in a:
		R=J.stem;c=g.get(R,0)+1;g[R]=c;E=R if c==1 else f"{R}-{c}";S=X/f"llama-swap-{sanitize_log_stem(E)}.log";B(f"Log file for '{E}': {S}");F=Z
		if F is _A and A.gpu_layer_autodetect:
			h=read_gguf_block_count(J)
			if h:F=str(h+1);B(f"Auto n-gpu-layers for '{E}': {F} (block_count + 1)")
			else:F=DEFAULT_N_GPU_LAYERS;B(f"Could not read block_count for '{E}'; using n-gpu-layers={F}.")
		if F is not _A:B(f"Using n-gpu-layers for '{E}': {F}")
		T=_A;U=_A
		if M is not _A and N is not _A:
			if M==BATCH_AUTO or N==BATCH_AUTO:
				if L is _A:L=detect_hardware_profile(B);p=f", {L.vram_gb:.1f} GB VRAM"if L.vram_gb else'';B(f"Auto batch profile: {L.name}{p}")
			try:T,U=resolve_batch_settings(M,N,A.allow_equal_batch,L,J,B)
			except ValueError as Y:print(f"ERROR: {Y} for model '{E}'",file=sys.stderr);return 2
		i=template_supports_thinking(J);B(f"Adding model '{E}' (thinking optional: {"yes"if i else"no"}).")
		if i:O.append((E,[f"  {yaml_key(E)}:",f"    cmd: {build_cmd(A.llama_server,J,_F,A.ctx_size,A.flash_attn,A.cache_type_k,A.cache_type_v,F,A.mmap,T,U,S)}"]));O.append((f"{E}-thinking",[f"  {yaml_key(E)}-thinking:",f"    cmd: {build_cmd(A.llama_server,J,_C,A.ctx_size,A.flash_attn,A.cache_type_k,A.cache_type_v,F,A.mmap,T,U,S)}"]))
		else:O.append((E,[f"  {yaml_key(E)}:",f"    cmd: {build_cmd(A.llama_server,J,_A,A.ctx_size,A.flash_attn,A.cache_type_k,A.cache_type_v,F,A.mmap,T,U,S)}"]))
	d=[]
	for(A0,e)in O:d.extend(e);d.append('')
	q=o+['']+d;j=W.join(q).rstrip()+W
	if A.output=='-':sys.stdout.write(j);B('Wrote config to stdout.');return 0
	I=Path(A.output).expanduser()
	if I.exists():
		C=I.read_text(encoding=_B).splitlines();G,H=find_models_block(C)
		if G is _A:
			if C and C[-1].strip():C.append('')
			C.append(_D);G=len(C)-1;H=len(C)
		if G is not _A and A.prune_missing:
			r=parse_models_entries(C,G,H);s={A for(A,B)in O};k=[_C]*len(C);V=0
			for(t,u,v)in r:
				if t in s:continue
				for w in range(u,v):k[w]=_F
				V+=1
			if V:C=[B for(A,B)in enumerate(C)if k[A]];G,H=find_models_block(C);B(f"Pruned {V} missing model entr{"y"if V==1 else n}.")
		G,H=find_models_block(C)
		if G is _A:
			if C and C[-1].strip():C.append('')
			C.append(_D);G=len(C)-1;H=len(C)
		x=existing_model_keys(C,G,H);K=[];f=0
		for(y,e)in O:
			if y in x:continue
			if K:K.append('')
			K.extend(e);f+=1
		if not K:B(f"No new entries to add; leaving config unchanged: {I}");return 0
		if H>G+1 and C[H-1].strip():K=['']+K
		C[H:H]=K;z=W.join(C).rstrip()+W;I.write_text(z,encoding=_B);B(f"Updated config at: {I} (added {f} entr{"y"if f==1 else n}).")
	else:I.parent.mkdir(parents=_C,exist_ok=_C);I.write_text(j,encoding=_B);B(f"Wrote new config to: {I}")
	return 0
if __name__=='__main__':raise SystemExit(main())