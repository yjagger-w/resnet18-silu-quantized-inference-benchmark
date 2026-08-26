"""Validate each embedded ORT piecewise site on its exact ORT SiLU input."""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort,torch,torchvision
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from silu_benchmark.backends import discover_silu_patterns,load_site_spec_manifest,rewrite_silu_piecewise_model,save_rewrite_result
from silu_benchmark.calibration_manifest import sha256_file
from silu_benchmark.data import cifar10_transform
from silu_benchmark.quantization import piecewise_dequantize,piecewise_quantize
def debug(model,names):
 m=onnx.shape_inference.infer_shapes(copy.deepcopy(model));known={v.name:v for v in [*m.graph.input,*m.graph.output,*m.graph.value_info]}
 for n in names:m.graph.output.append(copy.deepcopy(known[n]))
 return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,default=Path('artifacts/onnx/resnet18_silu_fp32.onnx'));p.add_argument('--manifest',type=Path,default=Path('configs/calibration/resnet18_silu_piecewise_v06_ort_cpu.json'));p.add_argument('--data-root',type=Path,default=Path('data'));p.add_argument('--output',type=Path,default=Path('artifacts/onnx/resnet18_silu_piecewise_ort_cpu.onnx'));p.add_argument('--report',type=Path,default=Path('results/silu_piecewise_ort_native_validation.json'));p.add_argument('--index',type=int,default=123);a=p.parse_args()
 payload=json.loads(a.manifest.read_text());
 if payload.get('calibration_backend')!='onnxruntime' or payload.get('execution_provider')!='CPUExecutionProvider':raise ValueError('manifest is not ORT CPU provenance')
 if payload['baseline_onnx_model']['sha256']!=sha256_file(a.baseline):raise ValueError('baseline ONNX digest mismatch')
 base=onnx.load(str(a.baseline));sites=discover_silu_patterns(base);specs=load_site_spec_manifest(a.manifest)
 if [x['site_id'] for x in payload['onnx_site_order']]!=[s.site_id for s in sites] or set(specs)!={s.site_id for s in sites}:raise ValueError('ORT manifest site order/count mismatch')
 result=rewrite_silu_piecewise_model(base,specs);save_rewrite_result(result,a.output);pre=[f"silu_piecewise_{s.site_id.replace('.','_')}_silu_output" for s in sites];codes=[result.inserted_outputs[s.site_id]['quantized_codes_tensor'] for s in sites];deq=[result.inserted_outputs[s.site_id]['dequantized_output_tensor'] for s in sites]
 image,_=torchvision.datasets.CIFAR10(a.data_root,train=False,download=False,transform=cifar10_transform())[a.index];x=image.unsqueeze(0).numpy();m=debug(result.model,[*pre,*codes,*deq]);session=ort.InferenceSession(m.SerializeToString(),providers=['CPUExecutionProvider']);one=session.run(None,{'images':x});two=session.run(None,{'images':x});names=[o.name for o in m.graph.output];d={n:v for n,v in zip(names[1:],one[1:])}; rows=[]
 for s,inp,c,q in zip(sites,pre,codes,deq):
  pyc=np.asarray(piecewise_quantize(d[inp],specs[s.site_id]),dtype=np.uint8);pyd=np.asarray(piecewise_dequantize(pyc,specs[s.site_id]),dtype=np.float32);err=np.abs(pyd-d[q]);rows.append({'site_id':s.site_id,'code_mismatch_count':int(np.count_nonzero(pyc-d[c])),'dequantized_max_error':float(err.max()),'dequantized_mean_error':float(err.mean())})
 report={'baseline_sha256':sha256_file(a.baseline),'rewritten_sha256':sha256_file(a.output),'manifest_sha256':sha256_file(a.manifest),'provider':'CPUExecutionProvider','site_count':len(sites),'remaining_silu_patterns':len(discover_silu_patterns(result.model)),'checker':True,'sites':rows,'total_code_mismatch_count':sum(r['code_mismatch_count'] for r in rows),'max_dequantized_error':max(r['dequantized_max_error'] for r in rows),'repeated_run_final_max_error':float(np.max(np.abs(one[0]-two[0]))),'repeated_run_site_code_exact':all(np.array_equal(a1,a2) for a1,a2 in zip(one[1+len(pre):1+len(pre)+len(codes)],two[1+len(pre):1+len(pre)+len(codes)]))};a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
