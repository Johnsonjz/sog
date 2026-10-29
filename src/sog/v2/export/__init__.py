"""``torch.export`` / AOTInductor helpers for SOG artifacts.

Two deployment chains are supported (see :mod:`sog.v2.export.artifacts`):
Mode A = pair + kspace + fix (energy-only / response / combined artifacts),
Mode B = full-LR single graph (``sog_full_lr``).
"""

from .abi import (
    COMM_TENSOR_NAMES,
    ENERGY_ONLY_OUTPUT_KEYS,
    FULL_LR_OUTPUT_KEYS,
    GRAPH_TENSOR_NAMES,
    RESPONSE_OUTPUT_KEYS,
    SINGLE_RANK_FULL_LR_INPUTS,
    WITH_COMM_FULL_LR_INPUTS,
    EnergyOnlyTraceable,
    FullLRTraceable,
    ResponseTraceable,
    make_coord_box_samples,
    validate_output_keys,
)
from .aoti import (
    check_toolchain,
    compile_artifact,
    compile_and_package,
    default_inductor_configs,
    filter_inductor_configs,
    load_package,
)
from .artifacts import (
    ARTIFACT_NAMES,
    PT2_EXTRA_PREFIX,
    build_metadata,
    install_artifacts,
    read_metadata,
)
from .trace import (
    ExportedArtifact,
    export_program,
    output_keys_of,
    strip_shape_assertions,
    trace_and_export,
    trace_with_make_fx,
)

__all__ = [
    "COMM_TENSOR_NAMES",
    "GRAPH_TENSOR_NAMES",
    "SINGLE_RANK_FULL_LR_INPUTS",
    "WITH_COMM_FULL_LR_INPUTS",
    "FULL_LR_OUTPUT_KEYS",
    "ENERGY_ONLY_OUTPUT_KEYS",
    "RESPONSE_OUTPUT_KEYS",
    "FullLRTraceable",
    "EnergyOnlyTraceable",
    "ResponseTraceable",
    "make_coord_box_samples",
    "validate_output_keys",
    "ExportedArtifact",
    "trace_with_make_fx",
    "export_program",
    "trace_and_export",
    "strip_shape_assertions",
    "output_keys_of",
    "default_inductor_configs",
    "filter_inductor_configs",
    "compile_artifact",
    "compile_and_package",
    "load_package",
    "check_toolchain",
    "PT2_EXTRA_PREFIX",
    "ARTIFACT_NAMES",
    "build_metadata",
    "install_artifacts",
    "read_metadata",
]
