"""Frozen v0.6.5 ORT CPU three-variant benchmark runner."""
from __future__ import annotations
import argparse,copy,csv,ctypes,hashlib,json,math,os,platform,subprocess,sys,tempfile,time,traceback,uuid
from pathlib import Path
from typing import Any,Callable,Mapping,Optional,Sequence
import numpy as np,onnx,onnxruntime as ort
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from silu_benchmark.backends import discover_silu_patterns,inspect_onnx_model,load_site_spec_manifest,rewrite_silu_piecewise_model,save_rewrite_result
from silu_benchmark.benchmark_data import load_cifar_batch,numpy_batches,build_standard_qdq
from silu_benchmark.calibration_manifest import load_manifest
VARIANTS=("fp32_ort_cpu","standard_static_qdq_ort_cpu","silu_piecewise_ort_reference_cpu")
REQUIRED_CONFIG={"schema_version","checkpoint","dataset","data_root","evaluation_samples","calibration_manifest","calibration_samples","calibration_batch_size","seed","provider","evaluation_batch_size","latency_batch_size","latency_warmup_runs","latency_timed_runs","throughput_batch_size","throughput_warmup_runs","throughput_timed_runs","standard_qdq","output_directory"}
def q(values):
 values=np.asarray(values,dtype=np.float64)
 if not len(values):raise ValueError("at least one timing sample is required")
 return {"mean_ms":float(np.mean(values)*1e3),"p50_ms":float(np.percentile(values,50)*1e3),"p95_ms":float(np.percentile(values,95)*1e3),"min_ms":float(np.min(values)*1e3),"max_ms":float(np.max(values)*1e3),"std_ms":float(np.std(values)*1e3)}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def metadata(path):return path.with_suffix(path.suffix+'.v065.json')
def valid(path,fingerprint):
 try:
  sidecar=json.loads(metadata(path).read_text(encoding='utf-8'))
  return path.is_file() and sidecar.get('fingerprint')==fingerprint and sidecar.get('sha256')==digest(path)
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

def build_fingerprint(config:Mapping[str,Any],input_hashes:Mapping[str,str],runtime_version:str,smoke:bool=False)->str:
 payload={'config':dict(config),'inputs':dict(sorted(input_hashes.items())),'onnxruntime':runtime_version,'provider':'CPUExecutionProvider','runner_schema':'v0.6.5-ort-only-v1','smoke':smoke,'qdq_calibration_order':'first training samples, sequential'}
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
 ref=reference.astype(np.float64,copy=False).ravel();other=candidate.astype(np.float64,copy=False).ravel()
 if not ref.size or not np.all(np.isfinite(ref)) or not np.all(np.isfinite(other)):raise ValueError('semantic outputs must be nonempty and finite')
 # Elementwise float64 reductions implement the same cosine definition without BLAS.
 denominator=math.sqrt(float(np.sum(ref*ref)))*math.sqrt(float(np.sum(other*other)))
 cosine=float(np.sum(ref*other)/denominator) if denominator else (1.0 if np.array_equal(ref,other) else 0.0);delta=ref-other
 return {'mapping_status':'matched','shape':list(reference.shape),'mse':float(np.mean(delta*delta)),'mae':float(np.mean(np.abs(delta))),'max_absolute_error':float(np.max(np.abs(delta))),'cosine_similarity':cosine}

def unavailable_site(site_id:str,reason:str)->dict:return {'site_id':site_id,'mapping_status':'unavailable','mapping_reason':reason}

def semantic_activation_site_analysis(base, qdq, custom, manifest_path, images, work):
    """Run debug-only copies after timing; all inference stays on ORT CPU."""
    from silu_benchmark.benchmark_semantics import semantic_output_mappings, instrument_model

    baseline = onnx.load(str(base))
    piecewise = onnx.load(str(custom))
    manifest, _specs = load_manifest(manifest_path)
    fp32_mapping, custom_mapping = semantic_output_mappings(baseline, piecewise, manifest)
    work.mkdir(parents=True, exist_ok=False)
    collected = {}
    for label, model, mapping in (
        ("fp32", baseline, fp32_mapping), ("piecewise", piecewise, custom_mapping)
    ):
        debug_path = work / (label + ".debug.onnx")
        onnx.save(instrument_model(model, mapping), str(debug_path))
        session = ort.InferenceSession(str(debug_path), providers=["CPUExecutionProvider"])
        values = session.run(list(mapping.values()), {session.get_inputs()[0].name: images})
        collected[label] = dict(zip(mapping, values))
        del session
    session = ort.InferenceSession(str(qdq), providers=["CPUExecutionProvider"])
    if len(session.get_outputs()) != 1:
        raise ValueError("Standard QDQ must have a single final logits output")
    qdq_logits = session.run([session.get_outputs()[0].name],
                             {session.get_inputs()[0].name: images})[0]
    reason = ("Standard-QDQ has no verified Phase 3 provenance mapping after QDQ insertion; "
              "intermediate tensor correspondence is not proven.")
    rows = []
    for site_id in list(fp32_mapping)[:-1]:
        rows.append({
            "site_id": site_id,
            "fp32_vs_standard_static_qdq": unavailable_site(site_id, reason),
            "fp32_vs_silu_piecewise_ort_reference": {
                "site_id": site_id,
                **semantic_metrics(collected["fp32"][site_id], collected["piecewise"][site_id]),
                "mapping_reason": "ordered manifest and Phase 3 rewrite metadata verified",
            },
        })
    return {
        "title": "semantic activation-site analysis", "input_shape": list(images.shape),
        "site_count": len(rows), "ordered_site_ids": list(fp32_mapping)[:-1],
        "output_mappings": {"fp32": fp32_mapping, "piecewise": custom_mapping},
        "debug_runtime": {"debug_only_model_copies": True, "timed_models_instrumented": False},
        "sites": rows,
        "final_logits": {
            "fp32_vs_standard_static_qdq": semantic_metrics(collected["fp32"]["logits"], qdq_logits),
            "fp32_vs_silu_piecewise_ort_reference": semantic_metrics(
                collected["fp32"]["logits"], collected["piecewise"]["logits"]),
        },
    }

def variant_graph_facts(variant_id:str,path:Path)->dict:
 facts=inspect_onnx_model(path);hist=facts['operator_histogram']
 return {'variant_id':variant_id,'model_path':str(path.resolve()),'model_sha256':facts['sha256'],'model_size_bytes':path.stat().st_size,'conv_count':facts['conv_count'],'quantize_linear_nodes':hist.get('QuantizeLinear',0),'dequantize_linear_nodes':hist.get('DequantizeLinear',0),'baseline_silu_patterns':facts['baseline_silu_pattern_count'],'piecewise_sites':facts['inserted_piecewise_subgraph_count'],'float64_reference_arithmetic':any('FLOAT64' in item for item in facts['portability_considerations']),'facts':facts}

def output_path(config:Mapping[str,Any],smoke:bool)->Path:
 official=ROOT/config['output_directory'];return official.parent/(official.name+'_smoke') if smoke else official

def write_reports(stage, environment, result):
    atomic_json(stage / "environment.json", environment)
    atomic_json(stage / "benchmark_results.json", result)
    rows = []
    for row in result["variants"]:
        flat = {key: row[key] for key in ("variant_id", "accuracy", "correct", "samples")
                if key in row}
        flat.update({"latency_" + key: value for key, value in row["latency"].items()})
        flat.update({
            "throughput_images_per_second": row["throughput_images_per_second"],
            "model_size_bytes": row["graph_facts"]["model_size_bytes"],
            "memory_status": row["memory"]["status"],
            "memory_baseline_bytes": row["memory"].get("baseline_bytes"),
            "memory_peak_observed_bytes": row["memory"].get("peak_observed_bytes"),
            "memory_reason": row["memory"].get("reason"),
        })
        rows.append(flat)
    atomic_csv(stage / "benchmark_results.csv", rows)
    graph = stage / "graph_reports"
    graph.mkdir(parents=True, exist_ok=True)
    for row in result["variants"]:
        atomic_json(graph / (row["variant_id"] + ".json"), row["graph_facts"]["facts"])
    semantic = result["semantic_activation_site_analysis"]
    semantic_rows = []
    comparisons = (
        ("fp32_vs_standard_static_qdq", VARIANTS[1]),
        ("fp32_vs_silu_piecewise_ort_reference", VARIANTS[2]),
    )
    for site in semantic.get("sites", []) + [
        {"site_id": "logits", **semantic.get("final_logits", {})}
    ]:
        for key, variant in comparisons:
            if key in site:
                values = site[key]
                semantic_rows.append({
                    "variant_id": variant, "site_id": site["site_id"], **values,
                    "shape": json.dumps(values.get("shape")) if "shape" in values else "",
                })
    if semantic_rows:
        atomic_csv(stage / "semantic_activation_sites.csv", semantic_rows)
    lines = [
        "# v0.6.5 ORT CPU benchmark", "", "## Completion status", "",
        "**" + result["completion_status"] + "**", "",
        "Smoke results are development validation only and are not official CIFAR-10 results."
        if result.get("smoke", result["completion_status"].startswith("smoke")) else "Official protocol completed.",
        "", "All accuracy and semantic outputs use ONNX Runtime CPU. No PyTorch/ORT equality is claimed.",
        "", "## Frozen protocol", "", "Fingerprint: `" + result["fingerprint"] + "`", "",
        "```json", json.dumps(result.get("frozen_config", {}), indent=2), "```", "",
        "Effective settings (including smoke overrides):", "",
        "```json", json.dumps(result.get("effective_settings", {}), indent=2), "```", "",
        "## Variant definitions and graph facts", "",
        "FP32 is the original float graph; Standard-QDQ uses static MinMax QUInt8 activations and per-channel QInt8 weights; the piecewise graph applies the committed SiLU reference rewrite.",
        "", "| Variant | Q nodes | DQ nodes | SiLU patterns | Piecewise sites | Size (bytes) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["variants"]:
        facts = row["graph_facts"]
        lines.append(f"| {row['variant_id']} | {facts['quantize_linear_nodes']} | {facts['dequantize_linear_nodes']} | {facts['baseline_silu_patterns']} | {facts['piecewise_sites']} | {facts['model_size_bytes']} |")
    lines += [
        "", "> **Comparability caveat:** Standard static QDQ is the future OpenVINO/QNN deployment baseline. The SiLU piecewise graph is an ONNX Runtime functional reference and is not a fair all-INT8 deployment comparison unless graph facts prove equal non-SiLU and weight-quantization treatment.",
        "", "## Accuracy and performance", "",
        "| Variant | Accuracy | p50 (ms) | p95 (ms) | Images/s | Memory baseline (bytes) | Peak observed (bytes) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["variants"]:
        lines.append(f"| {row['variant_id']} | {row['accuracy']:.4%} | {row['latency']['p50_ms']:.3f} | {row['latency'].get('p95_ms', 'N/A')} | {row['throughput_images_per_second']:.2f} | {row['memory'].get('baseline_bytes', 'unavailable')} | {row['memory'].get('peak_observed_bytes', 'unavailable')} |")
    lines += [
        "", "Memory is sampled process RSS/working set, not ORT tensor-allocator memory. Timing uses uninstrumented models.",
        "", "## Semantic activation-site analysis", "",
        f"{semantic['site_count']} ordered SiLU sites plus final logits; analysis input shape: {semantic.get('input_shape', 'fixture')}.",
        "Standard-QDQ sites without verified provenance are unavailable; names alone are not proof.",
        "", "| Comparison variant | Site | Status | MSE | MAE | Max abs | Cosine |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in semantic_rows:
        lines.append(f"| {row['variant_id']} | {row['site_id']} | {row['mapping_status']} | {row.get('mse', 'N/A')} | {row.get('mae', 'N/A')} | {row.get('max_absolute_error', 'N/A')} | {row.get('cosine_similarity', 'N/A')} |")
    lines += ["", "Per-site shapes and mapping reasons are included in JSON and semantic_activation_sites.csv.",
              "", "## Artifact SHA-256", ""]
    for name, value in result["artifact_sha256"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines += ["", "## Environment", "", "```json",
              json.dumps(environment, indent=2), "```"]
    atomic_text(stage / "benchmark_report.md", "\n".join(lines) + "\n")

def publish_stage(stage:Path,destination:Path)->None:
 status=json.loads((stage/'run_status.json').read_text(encoding='utf-8'))
 if status.get('status')!='success':raise RuntimeError('partial-result not promoted to official-result')
 if destination.exists():raise FileExistsError(f'official output path already exists: {destination}')
 stage.replace(destination)
def assert_ort_only():
    loaded = [name for name in sys.modules if name == "torch" or name.startswith("torch.")
              or name == "torchvision" or name.startswith("torchvision.")]
    if loaded:
        raise RuntimeError("ORT-only benchmark unexpectedly loaded Torch: " + loaded[0])
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").lower() not in ("", "false", "0"):
        raise RuntimeError("Remove the inherited KMP_DUPLICATE_LIB_OK override before benchmarking")


def effective_settings(config, smoke):
    return {
        "evaluation_samples": 128 if smoke else config["evaluation_samples"],
        "qdq_calibration_samples": config["calibration_batch_size"] if smoke else config["calibration_samples"],
        "latency_warmups": 2 if smoke else config["latency_warmup_runs"],
        "latency_runs": 5 if smoke else config["latency_timed_runs"],
        "throughput_warmups": 1 if smoke else config["throughput_warmup_runs"],
        "throughput_runs": 3 if smoke else config["throughput_timed_runs"],
        "semantic_samples": 1 if smoke else config["evaluation_batch_size"],
    }


def setup_run_paths(config, smoke):
    """Reserve a NEW staging directory; never touch an earlier attempt."""
    destination = output_path(config, smoke)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if smoke and destination.exists():
        destination = destination.with_name(destination.name + "-" + uuid.uuid4().hex[:12])
    stage = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=str(destination.parent)))
    return stage, destination


def update_status(stage, **fields):
    path = stage / "run_status.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(fields)
    atomic_json(path, payload)


def record_failure(stage, error, **fields):
    update_status(stage, status="failed", error=str(error), **fields)


def supervise_worker(command, stage, destination):
    """Catch both Python exceptions and native worker exits, including abort/exit(3)."""
    process = None
    log_path = stage / "worker.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
            update_status(stage, worker_pid=process.pid)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
            process.stdout.close()
        status = json.loads((stage / "run_status.json").read_text(encoding="utf-8"))
        if returncode != 0:
            record_failure(
                stage, status.get("error", "Benchmark worker exited abnormally"),
                worker_exit_code=returncode, log_path="worker.log",
                diagnostic_tail=log_path.read_text(encoding="utf-8")[-8000:],
                action="Inspect worker.log and the last recorded phase; fix the cause and rerun --smoke. No result was promoted.",
            )
            return returncode if returncode > 0 else 1
        if status.get("status") != "reports_ready":
            raise RuntimeError("Worker exited without completing all reports")
        required = ["environment.json", "benchmark_results.json", "benchmark_results.csv",
                    "benchmark_report.md", "semantic_activation_sites.csv"]
        if not all((stage / name).is_file() for name in required):
            raise RuntimeError("Worker report set is incomplete")
        update_status(stage, status="success", phase="complete", worker_exit_code=0)
        publish_stage(stage, destination)
        print(json.dumps({"status": status["completion_status"], "output": str(destination)}, indent=2), flush=True)
        return 0
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        record_failure(stage, exc, exception_type=type(exc).__name__, traceback=traceback.format_exc(),
                       action="Inspect worker.log, resolve the error, and rerun; existing results are preserved.")
        return 1


def ensure_artifact(path, fingerprint, resume, force_rebuild, builder, stage):
    if not needs_rebuild(path, fingerprint, resume, force_rebuild):
        print("Reusing validated artifact: " + path.name, flush=True)
        return
    temporary = stage / (path.stem + ".build.onnx")
    builder(temporary)
    onnx.checker.check_model(onnx.load(str(temporary)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(path)
    atomic_json(metadata(path), {
        "fingerprint": fingerprint, "sha256": digest(path),
        "provider": "CPUExecutionProvider", "onnxruntime": ort.__version__,
    })


def run_worker(request, stage):
    assert_ort_only()
    config = request["config"]
    smoke = request["smoke"]
    settings = effective_settings(config, smoke)
    fingerprint = request["fingerprint"]
    base = ROOT / "artifacts/onnx/resnet18_silu_fp32.onnx"
    suffix = "_smoke" if smoke else ""
    qdq = ROOT / ("artifacts/int8/resnet18_silu_int8_v065" + suffix + ".onnx")
    custom = ROOT / ("artifacts/onnx/resnet18_silu_piecewise_v065" + suffix + ".onnx")
    manifest_path = ROOT / config["calibration_manifest"]
    manifest, _specs = load_manifest(manifest_path)
    if manifest["baseline_onnx_model"]["sha256"] != request["input_hashes"]["base_onnx"]:
        raise ValueError("Committed manifest does not match the baseline ONNX SHA-256")
    if manifest["checkpoint"]["sha256"] != request["input_hashes"]["checkpoint"]:
        raise ValueError("Committed manifest does not match the checkpoint SHA-256")

    def phase(name):
        update_status(stage, status="running", phase=name)
        print("v0.6.5 phase: " + name, flush=True)

    # ORT's calibration instrumentation uses tempfile internally. Keep its new
    # temporary models under this generated attempt, not the workspace or system temp.
    runtime_tmp = stage / "runtime_tmp"
    runtime_tmp.mkdir()
    previous_tempdir = tempfile.tempdir
    tempfile.tempdir = str(runtime_tmp)
    try:
        phase("static QDQ")
        ensure_artifact(
            qdq, fingerprint, request["resume"], request["force_rebuild"],
            lambda target: build_standard_qdq(base, target, ROOT / config["data_root"],
                                              settings["qdq_calibration_samples"], config["calibration_batch_size"]),
            stage,
        )
    finally:
        tempfile.tempdir = previous_tempdir
    assert_ort_only()
    phase("custom rewrite")
    ensure_artifact(
        custom, fingerprint, request["resume"], request["force_rebuild"],
        lambda target: save_rewrite_result(
            rewrite_silu_piecewise_model(onnx.load(str(base)), load_site_spec_manifest(manifest_path)), target),
        stage,
    )
    images, labels = load_cifar_batch(ROOT / config["data_root"], "test_batch")
    indices = select_sample_indices(len(images), config["evaluation_samples"], smoke)
    first_images, _ = next(numpy_batches(images, labels, indices, config["evaluation_batch_size"]))
    throughput_images = first_images[:config["throughput_batch_size"]]
    if len(throughput_images) != config["throughput_batch_size"]:
        raise ValueError("evaluation batch is smaller than the throughput batch size")
    variants = []
    for variant_id, path in zip(VARIANTS, (base, qdq, custom)):
        phase(variant_id)
        # Never benchmark instrumented graphs: exposed outputs can inhibit optimization.
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        correct = total = 0
        for batch, batch_labels in numpy_batches(images, labels, indices, config["evaluation_batch_size"]):
            logits = session.run([output_name], {input_name: batch})[0]
            if not np.all(np.isfinite(logits)):
                raise ValueError("Non-finite logits from " + variant_id)
            correct += int((logits.argmax(1) == batch_labels).sum())
            total += len(batch_labels)
            if total % (10 * config["evaluation_batch_size"]) == 0:
                print(f"{variant_id}: evaluated {total}/{len(indices)}", flush=True)
        latency_times, memory = timed_runs(
            session, input_name, np.zeros((config["latency_batch_size"], 3, 32, 32), np.float32),
            settings["latency_warmups"], settings["latency_runs"],
        )
        throughput_times, throughput_memory = timed_runs(
            session, input_name, throughput_images, settings["throughput_warmups"], settings["throughput_runs"],
        )
        del session
        graph_facts = variant_graph_facts(variant_id, path)
        if not graph_facts["facts"]["checker"]["passed"] or not graph_facts["facts"]["runtime"]["run_passed"]:
            raise RuntimeError("Graph inspection failed for " + variant_id)
        variants.append({
            "variant_id": variant_id, "accuracy": correct / total, "correct": correct, "samples": total,
            "latency": q(latency_times),
            "throughput_images_per_second": float(config["throughput_batch_size"] / np.mean(throughput_times)),
            "memory": memory, "throughput_memory": throughput_memory, "graph_facts": graph_facts,
        })

    phase("semantic activation-site analysis")
    semantic = semantic_activation_site_analysis(
        base, qdq, custom, manifest_path, first_images[:settings["semantic_samples"]], stage / "debug_models",
    )
    assert_ort_only()
    options = ort.SessionOptions()
    environment = {
        "onnxruntime": ort.__version__, "onnx": onnx.__version__, "numpy": np.__version__,
        "provider": "CPUExecutionProvider", "available_providers": ort.get_available_providers(),
        "cpu_count": os.cpu_count(), "cpu": platform.processor(), "platform": platform.platform(),
        "python": sys.version, "python_executable": sys.executable, "torch_imported": False,
        "thread_settings": {
            "intra_op_num_threads": options.intra_op_num_threads,
            "inter_op_num_threads": options.inter_op_num_threads,
            "execution_mode": str(options.execution_mode),
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
        },
        "instrumented_models_used_for_timing": False,
    }
    result = {
        "schema_version": "benchmark-results/v0.6.5",
        "completion_status": "smoke-success" if smoke else "official-success", "smoke": smoke,
        "fingerprint": fingerprint, "frozen_config": config, "effective_settings": settings,
        "qdq_calibration_indices": list(range(settings["qdq_calibration_samples"])),
        "variants": variants, "semantic_activation_site_analysis": semantic,
        "artifact_sha256": {**request["input_hashes"], "standard_static_qdq": digest(qdq),
                            "silu_piecewise_reference": digest(custom)},
    }
    phase("writing reports")
    write_reports(stage, environment, result)
    update_status(stage, status="reports_ready", phase="awaiting supervisor",
                  completion_status=result["completion_status"])


def main():
    parser = argparse.ArgumentParser(description="Frozen v0.6.5 ORT CPU benchmark; supervised native execution.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_request:
        stage = args.worker_request.resolve().parent
        try:
            request = json.loads(args.worker_request.read_text(encoding="utf-8"))
            run_worker(request, stage)
            return 0
        except BaseException as exc:
            record_failure(stage, exc, exception_type=type(exc).__name__, traceback=traceback.format_exc())
            traceback.print_exc()
            return 1
    if args.config is None:
        parser.error("--config is required")
    config = validate_config(json.loads(args.config.read_text(encoding="utf-8")))
    stage, destination = setup_run_paths(config, args.smoke)
    update_status(stage, status="running", phase="initializing", smoke=args.smoke)
    print("Benchmark staging output: " + str(stage), flush=True)
    try:
        assert_ort_only()
        if destination.exists():
            raise FileExistsError("Refusing to overwrite existing output: " + str(destination))
        input_hashes = {
            "base_onnx": digest(ROOT / "artifacts/onnx/resnet18_silu_fp32.onnx"),
            "manifest": digest(ROOT / config["calibration_manifest"]),
            "checkpoint": digest(ROOT / config["checkpoint"]),
            "test_batch": digest(ROOT / config["data_root"] / "cifar-10-batches-py/test_batch"),
            "qdq_training_batch": digest(ROOT / config["data_root"] / "cifar-10-batches-py/data_batch_1"),
            "runner_source": digest(Path(__file__)),
            "data_source": digest(ROOT / "src/silu_benchmark/benchmark_data.py"),
            "semantic_source": digest(ROOT / "src/silu_benchmark/benchmark_semantics.py"),
        }
        fingerprint = build_fingerprint(config, input_hashes, ort.__version__, args.smoke)
        request = {
            "config": config, "smoke": args.smoke, "resume": args.resume,
            "force_rebuild": args.force_rebuild, "input_hashes": input_hashes, "fingerprint": fingerprint,
        }
        update_status(stage, fingerprint=fingerprint)
        request_path = stage / "worker_request.json"
        atomic_json(request_path, request)
        return supervise_worker(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-request", str(request_path)],
            stage, destination,
        )
    except BaseException as exc:
        record_failure(stage, exc, exception_type=type(exc).__name__, traceback=traceback.format_exc())
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
