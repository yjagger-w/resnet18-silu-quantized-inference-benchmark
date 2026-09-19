"""Functional backend reference implementations."""

from .onnx_piecewise import (
    PIECEWISE_ONNX_OPSET,
    PiecewiseSubgraphOutputs,
    build_piecewise_qdq_model,
    build_piecewise_qdq_subgraph,
    create_piecewise_ort_session,
)
from .onnx_model_rewrite import (
    SiLUPatternSite,
    bind_module_specs_to_sites,
    discover_silu_patterns,
    inspect_onnx_model,
    load_site_spec_manifest,
    rewrite_silu_piecewise_model,
    save_rewrite_result,
    write_inspection_report,
)
from .qnn_aihub_backend import (
    QaiHubUnavailableError,
    QnnAiHubBackend,
    build_numerical_audit,
    build_profile_summary,
    generate_audit_inputs,
    numerical_metrics,
    summarize_profile,
)

__all__ = [
    "PIECEWISE_ONNX_OPSET",
    "PiecewiseSubgraphOutputs",
    "build_piecewise_qdq_model",
    "build_piecewise_qdq_subgraph",
    "create_piecewise_ort_session",
    "SiLUPatternSite",
    "bind_module_specs_to_sites",
    "discover_silu_patterns",
    "inspect_onnx_model",
    "load_site_spec_manifest",
    "rewrite_silu_piecewise_model",
    "save_rewrite_result",
    "write_inspection_report",
    "QaiHubUnavailableError",
    "QnnAiHubBackend",
    "build_numerical_audit",
    "build_profile_summary",
    "generate_audit_inputs",
    "numerical_metrics",
    "summarize_profile",
]
