"""Frozen v0.6.5 ORT CPU three-variant benchmark runner."""
from __future__ import annotations
import argparse,csv,ctypes,hashlib,json,os,platform,shutil,subprocess,sys,tempfile,time,traceback
from pathlib import Path
from typing import Any,Callable,Mapping,Optional,Sequence
import numpy as np,onnx,onnxruntime as ort,torch,torchvision
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from silu_benchmark.backends import discover_silu_patterns,inspect_onnx_model,load_site_spec_manifest,rewrite_silu_piecewise_model,save_rewrite_result
from silu_benchmark.data import cifar10_transform
VARIANTS=("fp32_ort_cpu","standard_static_qdq_ort_cpu","silu_piecewise_ort_reference_cpu")
REQUIRED_CONFIG={"schema_version","checkpoint","dataset","data_root","evaluation_samples","calibration_manifest","calibration_samples","calibration_batch_size","seed","provider","evaluation_batch_size","latency_batch_size","latency_warmup_runs","latency_timed_runs","throughput_batch_size","throughput_warmup_runs","throughput_timed_runs","standard_qdq","output_directory"}
def q(values):
 values=np.asarray(values,dtype=np.float64)
 if not len(values):raise ValueError("at least one timing sample is required")
 return {"mean_ms":float(np.mean(values)*1e3),"p50_ms":float(np.percentile(values,50)*1e3),"p95_ms":float(np.percentile(values,95)*1e3),"min_ms":float(np.min(values)*1e3),"max_ms":float(np.max(values)*1e3),"std_ms":float(np.std(values)*1e3)}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def metadata(path):return path.with_suffix(path.suffix+'.v065.json')
def valid(path,fingerprint):
 try:return path.exists() and metadata(path).exists() and json.loads(metadata(path).read_text(encoding='utf-8')).get('fingerprint')==fingerprint
 except (OSError,json.JSONDecodeError):return False

def validate_config(config:Mapping[str,Any])->dict:
 missing=sorted(REQUIRED_CONFIG-set(config))
 if missing:raise ValueError('benchmark config missing: '+', '.join(missing))
 if config['schema_version']!='benchmark/v0.6.5':raise ValueError('unsupported benchmark schema version')
 if config['dataset']!='CIFAR-10' or config['calibration_samples']!=2560:raise ValueError('v0.6.5 requires CIFAR-10 and 2,560 calibration samples')
 if config['provider']!='CPUExecutionProvider':raise ValueError('v0.6.5 runner supports CPUExecutionProvider only')
 if config['evaluation_samples']!=10000:raise ValueError('official evaluation must use all 10,000 CIFAR-10 test samples')
 return dict(config)

def select_sample_indices(total:int,official_samples:int,smoke:bool)->list[int]:
 if total<official_samples:raise ValueError('CIFAR-10 test set is smaller than the frozen evaluation protocol')
 return list(range(min(128,total) if smoke else official_samples))

def build_fingerprint(config:Mapping[str,Any],input_hashes:Mapping[str,str],runtime_version:str)->str:
 payload={'config':dict(config),'inputs':dict(sorted(input_hashes.items())),'onnxruntime':runtime_version,'provider':'CPUExecutionProvider','runner_schema':'v0.6.5'}
 return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()

def needs_rebuild(path:Path,fingerprint:str,resume:bool,force_rebuild:bool)->bool:
 return bool(force_rebuild or not resume or not valid(path,fingerprint))

def atomic_text(path:Path,text:str)->None:
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',encoding='utf-8',newline='',dir=str(path.parent),delete=False) as handle:
  handle.write(text);temporary=Path(handle.name)
 temporary.replace(path)

def atomic_json(path:Path,payload:Any)->None:atomic_text(path,json.dumps(payload,indent=2,sort_keys=True)+'\n')
def atomic_csv(path:Path,rows:Sequence[Mapping[str,Any]])->None:
 if not rows:raise ValueError('cannot write an empty CSV report')
 fields=sorted({key for row in rows for key in row})
 with tempfile.NamedTemporaryFile('w',encoding='utf-8',newline='',dir=str(path.parent),delete=False) as handle:
  writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows);temporary=Path(handle.name)
 temporary.replace(path)

def process_memory_snapshot(query:Optional[Callable[[],int]]=None)->dict:
 if query is not None:
  try:return {'status':'available','bytes':int(query()),'source':'injected'}
  except Exception as exc:return {'status':'unavailable','reason':str(exc)}
 try:
  import psutil
  return {'status':'available','bytes':int(psutil.Process().memory_info().rss),'source':'psutil RSS'}
 except ImportError:pass
 except Exception as exc:return {'status':'unavailable','reason':'psutil RSS query failed: '+str(exc)}
 if os.name=='nt':
  try:
   class Counters(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
   counters=Counters();counters.cb=ctypes.sizeof(Counters)
   if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(counters),counters.cb):raise ctypes.WinError()
   return {'status':'available','bytes':int(counters.WorkingSetSize),'source':'Windows working set'}
  except Exception as exc:return {'status':'unavailable','reason':'Windows working-set query failed: '+str(exc)}
 return {'status':'unavailable','reason':'psutil is unavailable and no platform working-set query is implemented'}

def timed_runs(session:ort.InferenceSession,input_name:str,values:np.ndarray,warmups:int,runs:int,memory_query:Optional[Callable[[],int]]=None)->tuple[list[float],dict]:
 output_name=session.get_outputs()[0].name
 for _ in range(warmups):session.run([output_name],{input_name:values})
 baseline=process_memory_snapshot(memory_query);observed=[];times=[]
 for _ in range(runs):
  start=time.perf_counter_ns();session.run([output_name],{input_name:values});times.append((time.perf_counter_ns()-start)/1e9)
  sample=process_memory_snapshot(memory_query)
  if sample['status']=='available':observed.append(sample['bytes'])
 if baseline['status']!='available':return times,{'status':'unavailable','reason':baseline['reason'],'scope':'process-level RSS/working set, not ORT tensor-allocator memory'}
 if not observed:return times,{'status':'unavailable','reason':'no process-memory samples were available during timed execution','scope':'process-level RSS/working set, not ORT tensor-allocator memory'}
 return times,{'status':'available','baseline_bytes':baseline['bytes'],'peak_observed_bytes':max(observed),'source':baseline.get('source'),'scope':'process-level RSS/working set, not ORT tensor-allocator memory'}

def semantic_metrics(reference:np.ndarray,candidate:np.ndarray)->dict:
 if reference.shape!=candidate.shape:return {'mapping_status':'unavailable','mapping_reason':f'incompatible shapes: FP32 {list(reference.shape)} vs candidate {list(candidate.shape)}'}
 ref=reference.astype(np.float64,copy=False).ravel();other=candidate.astype(np.float64,copy=False).ravel();denominator=float(np.linalg.norm(ref)*np.linalg.norm(other));cosine=float(np.dot(ref,other)/denominator) if denominator else (1.0 if np.array_equal(ref,other) else 0.0);delta=ref-other
 return {'mapping_status':'matched','shape':list(reference.shape),'mse':float(np.mean(delta*delta)),'mae':float(np.mean(np.abs(delta))),'max_absolute_error':float(np.max(np.abs(delta))),'cosine_similarity':cosine}

def unavailable_site(site_id:str,reason:str)->dict:return {'site_id':site_id,'mapping_status':'unavailable','mapping_reason':reason}

def add_debug_outputs(source:Path,tensors:Sequence[str],destination:Path)->None:
 model=onnx.load(str(source))
 try:model=onnx.shape_inference.infer_shapes(model)
 except (onnx.shape_inference.InferenceError,ValueError):pass
 known={item.name:item for item in list(model.graph.value_info)+list(model.graph.input)+list(model.graph.output)};existing={item.name for item in model.graph.output}
 for tensor in tensors:
  if tensor not in existing:
   value=known.get(tensor) or onnx.helper.make_tensor_value_info(tensor,onnx.TensorProto.FLOAT,[None,None,None,None])
   model.graph.output.append(value);existing.add(tensor)
 onnx.checker.check_model(model);onnx.save(model,str(destination))

def run_debug_outputs(source:Path,tensors:Sequence[str],images:np.ndarray,work:Path)->dict[str,np.ndarray]:
 work.mkdir(parents=True,exist_ok=True);debug=work/(source.stem+'.debug.onnx');add_debug_outputs(source,tensors,debug)
 inputs=work/(source.stem+'.debug-input.npy');outputs=work/(source.stem+'.debug-output.npz');np.save(inputs,images)
 program="import sys,numpy as np,onnxruntime as ort; s=ort.InferenceSession(sys.argv[1],providers=['CPUExecutionProvider']); r=s.run(None,{s.get_inputs()[0].name:np.load(sys.argv[2])}); np.savez_compressed(sys.argv[3],*r)"
 environment=dict(os.environ);environment['KMP_DUPLICATE_LIB_OK']='TRUE'
 subprocess.run([sys.executable,'-c',program,str(debug),str(inputs),str(outputs)],check=True,env=environment)
 with np.load(outputs) as archive:return {item.name:archive[f'arr_{index}'] for index,item in enumerate(onnx.load(str(debug)).graph.output)}

def semantic_activation_site_analysis(base:Path,qdq:Path,custom:Path,images:np.ndarray,work:Path)->dict:
 sites=discover_silu_patterns(onnx.load(str(base)));site_tensors=[site.output_tensor for site in sites];fp32=run_debug_outputs(base,site_tensors,images,work);custom_outputs=run_debug_outputs(custom,site_tensors,images,work)
 qdq_session=ort.InferenceSession(str(qdq),providers=['CPUExecutionProvider']);qdq_logits=qdq_session.run(None,{qdq_session.get_inputs()[0].name:images})[0]
 custom_session=ort.InferenceSession(str(custom),providers=['CPUExecutionProvider']);custom_logits=custom_session.run(None,{custom_session.get_inputs()[0].name:images})[0]
 fp32_logits=fp32[next(iter(onnx.load(str(base)).graph.output)).name]
 qdq_reason='unavailable: ONNX Runtime static QDQ does not retain Phase 3 site identities or provenance mappings; tensor correspondence is not proven.'
 rows=[]
 for site in sites:
  reference=fp32[site.output_tensor];custom_row={'site_id':site.site_id,**semantic_metrics(reference,custom_outputs[site.output_tensor]),'mapping_reason':'proven by Phase 3 rewrite metadata and preserved output tensor identity'}
  rows.append({'site_id':site.site_id,'fp32_vs_standard_static_qdq':unavailable_site(site.site_id,qdq_reason),'fp32_vs_silu_piecewise_ort_reference':custom_row})
 return {'title':'semantic activation-site analysis','input_shape':list(images.shape),'site_count':len(sites),'debug_runtime':{'isolated_processes':True,'KMP_DUPLICATE_LIB_OK':'TRUE (debug helpers only; timed benchmark sessions are unchanged)'},'sites':rows,'final_logits':{'fp32_vs_standard_static_qdq':semantic_metrics(fp32_logits,qdq_logits),'fp32_vs_silu_piecewise_ort_reference':semantic_metrics(fp32_logits,custom_logits)}}

def semantic_from_session_outputs(sites,fp32,qdq,custom,logit_name,images):
 qdq_reason='unavailable: ONNX Runtime static QDQ does not retain Phase 3 site identities or provenance mappings; tensor correspondence is not proven.';rows=[]
 for site in sites:
  reference=fp32[site.output_tensor];rows.append({'site_id':site.site_id,'fp32_vs_standard_static_qdq':unavailable_site(site.site_id,qdq_reason),'fp32_vs_silu_piecewise_ort_reference':{'site_id':site.site_id,**semantic_metrics(reference,custom[site.output_tensor]),'mapping_reason':'proven by Phase 3 rewrite metadata and preserved output tensor identity'}})
 return {'title':'semantic activation-site analysis','input_shape':list(images.shape),'site_count':len(sites),'debug_runtime':{'debug_only_model_copies':True,'timed_sessions_requested_logits_only':True},'sites':rows,'final_logits':{'fp32_vs_standard_static_qdq':semantic_metrics(fp32[logit_name],qdq[logit_name]),'fp32_vs_silu_piecewise_ort_reference':semantic_metrics(fp32[logit_name],custom[logit_name])}}

def variant_graph_facts(variant_id:str,path:Path)->dict:
 facts=inspect_onnx_model(path);hist=facts['operator_histogram']
 return {'variant_id':variant_id,'model_path':str(path.resolve()),'model_sha256':facts['sha256'],'model_size_bytes':path.stat().st_size,'conv_count':facts['conv_count'],'quantize_linear_nodes':hist.get('QuantizeLinear',0),'dequantize_linear_nodes':hist.get('DequantizeLinear',0),'baseline_silu_patterns':facts['baseline_silu_pattern_count'],'piecewise_sites':facts['inserted_piecewise_subgraph_count'],'float64_reference_arithmetic':any('FLOAT64' in item for item in facts['portability_considerations']),'facts':facts}

def output_path(config:Mapping[str,Any],smoke:bool)->Path:
 official=ROOT/config['output_directory'];return official.parent/(official.name+'_smoke') if smoke else official

def write_reports(stage:Path,environment:Mapping[str,Any],result:Mapping[str,Any])->None:
 atomic_json(stage/'environment.json',environment);atomic_json(stage/'benchmark_results.json',result)
 rows=[]
 for row in result['variants']:
  flat={key:value for key,value in row.items() if key not in {'graph_facts','memory'}};flat.update({'memory_status':row['memory']['status'],'memory_peak_observed_bytes':row['memory'].get('peak_observed_bytes')});rows.append(flat)
 atomic_csv(stage/'benchmark_results.csv',rows)
 graph=stage/'graph_reports';graph.mkdir(parents=True,exist_ok=True)
 for row in result['variants']:atomic_json(graph/(row['variant_id']+'.json'),row['graph_facts']['facts'])
 lines=['# v0.6.5 ORT CPU benchmark','','## Completion status','',f"**{result['completion_status']}**",'', '## Frozen protocol','',f"Fingerprint: `{result['fingerprint']}`",'', '## Variant definitions and graph facts','', '| Variant | Q nodes | DQ nodes | SiLU patterns | Piecewise sites | Size (bytes) |','|---|---:|---:|---:|---:|---:|']
 for row in result['variants']:
  facts=row['graph_facts'];lines.append(f"| {row['variant_id']} | {facts['quantize_linear_nodes']} | {facts['dequantize_linear_nodes']} | {facts['baseline_silu_patterns']} | {facts['piecewise_sites']} | {facts['model_size_bytes']} |")
 lines+=['','> **Comparability caveat:** Standard static QDQ is the future OpenVINO/QNN deployment baseline. The SiLU piecewise graph is an ONNX Runtime functional reference and is not a fair all-INT8 deployment comparison unless graph facts prove equal non-SiLU and weight-quantization treatment.','','## Accuracy and performance','', '| Variant | Accuracy | p50 latency (ms) | Throughput (images/s) | Process-memory peak |','|---|---:|---:|---:|---:|']
 for row in result['variants']:lines.append(f"| {row['variant_id']} | {row['accuracy']:.4%} | {row['latency']['p50_ms']:.3f} | {row['throughput_images_per_second']:.2f} | {row['memory'].get('peak_observed_bytes','unavailable')} |")
 lines+=['','## Semantic activation-site analysis','',f"{result['semantic_activation_site_analysis']['site_count']} Phase 3 SiLU output sites were examined; Standard-QDQ mappings are marked unavailable unless proven. Final logits are included when shapes are compatible.",'','## Artifact SHA-256','']
 for name,value in result['artifact_sha256'].items():lines.append(f'- `{name}`: `{value}`')
 lines+=['','## Environment','',f"Provider: `{environment['provider']}`; ONNX Runtime: `{environment['onnxruntime']}`; CPU count: `{environment['cpu_count']}`; thread settings: `{environment['thread_settings']}`."]
 atomic_text(stage/'benchmark_report.md','\n'.join(lines)+'\n')

def publish_stage(stage:Path,destination:Path)->None:
 status=json.loads((stage/'run_status.json').read_text(encoding='utf-8'))
 if status.get('status')!='success':raise RuntimeError('partial-result not promoted to official-result')
 if destination.exists():raise FileExistsError(f'official output path already exists: {destination}')
 stage.replace(destination)
def main():
 parser=argparse.ArgumentParser(description='Frozen v0.6.5 ORT CPU three-variant benchmark runner.')
 parser.add_argument('--config',type=Path,required=True);parser.add_argument('--resume',action='store_true');parser.add_argument('--force-rebuild',action='store_true');parser.add_argument('--smoke',action='store_true')
 args=parser.parse_args();config=validate_config(json.loads(args.config.read_text(encoding='utf-8')))
 torch.manual_seed(config['seed']);np.random.seed(config['seed'])
 base=ROOT/'artifacts/onnx/resnet18_silu_fp32.onnx';suffix='_smoke' if args.smoke else ''
 qdq=ROOT/f'artifacts/int8/resnet18_silu_int8_v065{suffix}.onnx';custom=ROOT/f'artifacts/onnx/resnet18_silu_piecewise_v065{suffix}.onnx'
 inputs={'base_onnx':digest(base),'manifest':digest(ROOT/config['calibration_manifest']),'checkpoint':digest(ROOT/config['checkpoint'])};fingerprint=build_fingerprint(config,inputs,ort.__version__)
 destination=output_path(config,args.smoke);stage=destination.parent/(destination.name+'.partial-'+fingerprint[:12])
 if destination.exists():raise FileExistsError(f'refusing to overwrite existing completed output: {destination}')
 stage.mkdir(parents=True,exist_ok=True);atomic_json(stage/'run_status.json',{'status':'running','smoke':args.smoke,'fingerprint':fingerprint,'phase':'initializing'})
 try:
  print('v0.6.5 phase: static QDQ',flush=True)
  if needs_rebuild(qdq,fingerprint,args.resume,args.force_rebuild):
   calibration_batches=1 if args.smoke else config['calibration_samples']//config['calibration_batch_size']
   subprocess.run([sys.executable,'scripts/quantize_int8_static.py','--model-input',str(base),'--model-output',str(qdq),'--data-root',config['data_root'],'--calibration-batch-size',str(config['calibration_batch_size']),'--calibration-batches',str(calibration_batches),'--calibration-method','MinMax'],check=True)
   atomic_json(metadata(qdq),{'fingerprint':fingerprint,'provider':'CPUExecutionProvider','onnxruntime':ort.__version__,'smoke':args.smoke,'calibration_batches':calibration_batches})
  print('v0.6.5 phase: custom rewrite',flush=True)
  if needs_rebuild(custom,fingerprint,args.resume,args.force_rebuild):
   save_rewrite_result(rewrite_silu_piecewise_model(onnx.load(str(base)),load_site_spec_manifest(ROOT/config['calibration_manifest'])),custom)
   atomic_json(metadata(custom),{'fingerprint':fingerprint,'provider':'CPUExecutionProvider','onnxruntime':ort.__version__,'smoke':args.smoke})
  atomic_json(stage/'run_status.json',{'status':'running','smoke':args.smoke,'fingerprint':fingerprint,'phase':'evaluating'})
  dataset=torchvision.datasets.CIFAR10(ROOT/config['data_root'],train=False,download=False,transform=cifar10_transform())
  indices=select_sample_indices(len(dataset),config['evaluation_samples'],args.smoke);subset=torch.utils.data.Subset(dataset,indices);loader=torch.utils.data.DataLoader(subset,batch_size=config['evaluation_batch_size'],shuffle=False)
  semantic_images=next(iter(loader))[0].numpy().astype(np.float32);throughput_images=semantic_images[:config['throughput_batch_size']]
  if len(throughput_images)!=config['throughput_batch_size']:raise ValueError('test batch is smaller than frozen throughput batch size')
  sites=discover_silu_patterns(onnx.load(str(base)));debug_work=stage/'debug_models';base_debug=debug_work/'fp32.debug.onnx';custom_debug=debug_work/'piecewise.debug.onnx';site_tensors=[site.output_tensor for site in sites];add_debug_outputs(base,site_tensors,base_debug);add_debug_outputs(custom,site_tensors,custom_debug)
  variants=[]
  latency_warmups=2 if args.smoke else config['latency_warmup_runs'];latency_runs=5 if args.smoke else config['latency_timed_runs'];throughput_warmups=1 if args.smoke else config['throughput_warmup_runs'];throughput_runs=3 if args.smoke else config['throughput_timed_runs']
  sessions={}
  for variant_id,path,execution_path in zip(VARIANTS,(base,qdq,custom),(base_debug,qdq,custom_debug)):
   print('v0.6.5 phase: '+variant_id,flush=True);session=ort.InferenceSession(str(execution_path),providers=['CPUExecutionProvider']);input_name=session.get_inputs()[0].name;output_name=session.get_outputs()[0].name;correct=total=0
   for images,labels in loader:
    logits=session.run([output_name],{input_name:images.numpy().astype(np.float32)})[0];correct+=int((logits.argmax(1)==labels.numpy()).sum());total+=len(labels)
   latency_times,memory=timed_runs(session,input_name,np.zeros((config['latency_batch_size'],3,32,32),np.float32),latency_warmups,latency_runs)
   throughput_times,_=timed_runs(session,input_name,throughput_images,throughput_warmups,throughput_runs)
   variants.append({'variant_id':variant_id,'accuracy':correct/total,'correct':correct,'samples':total,'latency':q(latency_times),'throughput_images_per_second':float(config['throughput_batch_size']/np.mean(throughput_times)),'memory':memory,'graph_facts':variant_graph_facts(variant_id,path)})
   sessions[variant_id]=(session,input_name,output_name)
  print('v0.6.5 phase: semantic activation-site analysis',flush=True)
  outputs={}
  for variant_id,(session,input_name,_output_name) in sessions.items():outputs[variant_id]=dict(zip([item.name for item in session.get_outputs()],session.run(None,{input_name:semantic_images})))
  logit_name=sessions['fp32_ort_cpu'][2];semantic=semantic_from_session_outputs(sites,outputs['fp32_ort_cpu'],outputs['standard_static_qdq_ort_cpu'],outputs['silu_piecewise_ort_reference_cpu'],logit_name,semantic_images)
  environment={'onnxruntime':ort.__version__,'provider':'CPUExecutionProvider','available_providers':ort.get_available_providers(),'cpu_count':os.cpu_count(),'platform':platform.platform(),'python':sys.version,'thread_settings':{'OMP_NUM_THREADS':os.getenv('OMP_NUM_THREADS'),'ORT_INTRA_OP_NUM_THREADS':os.getenv('ORT_INTRA_OP_NUM_THREADS'),'ORT_INTER_OP_NUM_THREADS':os.getenv('ORT_INTER_OP_NUM_THREADS')}}
  result={'schema_version':'benchmark-results/v0.6.5','completion_status':'smoke-success' if args.smoke else 'official-success','smoke':args.smoke,'fingerprint':fingerprint,'frozen_config':config,'variants':variants,'semantic_activation_site_analysis':semantic,'artifact_sha256':{**inputs,'standard_static_qdq':digest(qdq),'silu_piecewise_reference':digest(custom)}}
  write_reports(stage,environment,result);shutil.rmtree(stage/'debug_models',ignore_errors=True);atomic_json(stage/'run_status.json',{'status':'success','smoke':args.smoke,'fingerprint':fingerprint,'completion_status':result['completion_status']});publish_stage(stage,destination);print(json.dumps({'status':result['completion_status'],'output':str(destination.resolve())},indent=2),flush=True)
 except Exception as exc:
  atomic_json(stage/'run_status.json',{'status':'failed','smoke':args.smoke,'fingerprint':fingerprint,'error':str(exc),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
