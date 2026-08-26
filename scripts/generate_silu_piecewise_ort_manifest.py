"""Generate a v0.6 manifest from actual ONNX Runtime SiLU pattern outputs."""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort, torch, torchvision
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from silu_benchmark.backends import discover_silu_patterns
from silu_benchmark.calibration_manifest import build_manifest, sha256_file, write_manifest_atomic
from silu_benchmark.data import cifar10_transform

def debug_model(model,names):
    model=onnx.shape_inference.infer_shapes(copy.deepcopy(model)); known={v.name:v for v in [*model.graph.input,*model.graph.output,*model.graph.value_info]}
    for name in names: model.graph.output.append(copy.deepcopy(known[name]))
    onnx.checker.check_model(model); return model

def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,default=Path('artifacts/onnx/resnet18_silu_fp32.onnx'));p.add_argument('--checkpoint',type=Path,default=Path('checkpoints/resnet18_cifar10.pth'));p.add_argument('--data-root',type=Path,default=Path('data'));p.add_argument('--batch-size',type=int,default=128);p.add_argument('--num-calibration-batches',type=int,default=20);p.add_argument('--seed',type=int,default=20260826);p.add_argument('--output',type=Path,default=Path('configs/calibration/resnet18_silu_piecewise_v06_ort_cpu.json'));a=p.parse_args()
 torch.manual_seed(a.seed);np.random.seed(a.seed); model=onnx.load(str(a.baseline));sites=discover_silu_patterns(model);names=[s.output_tensor for s in sites]; session=ort.InferenceSession(debug_model(model,names).SerializeToString(),providers=['CPUExecutionProvider'])
 ds=torchvision.datasets.CIFAR10(a.data_root,train=True,download=False,transform=cifar10_transform()); count=a.batch_size*a.num_calibration_batches;indices=torch.randperm(len(ds),generator=torch.Generator().manual_seed(a.seed))[:count].tolist();loader=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,indices),batch_size=a.batch_size,shuffle=False)
 values={s.site_id:[] for s in sites}
 for images,_ in loader:
  out=session.run(names,{'images':images.numpy()})
  for site,array in zip(sites,out):
   flat=array.reshape(-1);values[site.site_id].append(flat[np.linspace(0,flat.size-1,min(100,flat.size),dtype=np.int64)])
 metadata={'calibration_backend':'onnxruntime','execution_provider':'CPUExecutionProvider','onnx_version':onnx.__version__,'onnxruntime_version':ort.__version__,'baseline_onnx_model':{'logical_id':a.baseline.as_posix(),'sha256':sha256_file(a.baseline),'opset':int(model.opset_import[0].version)},'model_architecture':'ResNet18-SiLU-CIFAR10','checkpoint':{'logical_id':a.checkpoint.as_posix(),'sha256':sha256_file(a.checkpoint)},'calibration_dataset':{'logical_id':'CIFAR-10/train','data_root':a.data_root.as_posix(),'preprocessing':'ToTensor; Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010])'},'calibration':{'batch_size':a.batch_size,'num_batches':a.num_calibration_batches,'sample_count':count,'seed':a.seed,'sample_indices':indices},'onnx_site_order':[{'site_id':s.site_id,'output_tensor':s.output_tensor,'module_path':s.module_path,'invocation_index':s.call_index} for s in sites]}
 manifest=build_manifest(site_activations={k:np.concatenate(v) for k,v in values.items()},metadata=metadata);write_manifest_atomic(manifest,a.output);print(a.output);print(len(sites),min(x['vsplit'] for x in manifest['sites']),max(x['vsplit'] for x in manifest['sites']))
if __name__=='__main__':main()
