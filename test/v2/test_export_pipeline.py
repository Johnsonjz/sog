"""Export pipeline: make_fx -> torch.export -> artifact metadata / AOTI.

The fixture model (``test/fixtures/tiny_charge_model.py``) implements the
three traceable protocols; the physics is closed-form so the tests focus on
the artifact mechanics.
"""

import zipfile

import pytest
import torch

from sog.v2.export import (
    build_metadata,
    check_toolchain,
    compile_artifact,
    export_program,
    install_artifacts,
    load_package,
    output_keys_of,
    read_metadata,
)
from sog.v2.export.artifacts import ARTIFACT_NAMES, PT2_EXTRA_PREFIX

from _helpers import make_kernel
from tiny_charge_model import TinyChargeModel

pytestmark = pytest.mark.v2


@pytest.fixture(scope="module")
def model():
    return TinyChargeModel(make_kernel(), nk=24, n_dl=0.5, box_length=10.0)


def test_full_lr_trace_and_export(model):
    inputs = model.full_lr_sample_inputs()
    traced = model.trace_full_lr(*inputs)

    keys = output_keys_of(traced, inputs)
    assert keys == ("energy", "force", "virial"), keys
    with torch.no_grad():
        out = traced(*inputs)
    assert out["energy"].shape == (1,)
    assert out["force"].shape == (8, 3)
    assert out["virial"].shape == (1, 9)

    exported = export_program(traced, inputs)
    assert exported is not None
    gm = exported.module() if hasattr(exported, "module") else exported
    with torch.no_grad():
        got = gm(*inputs)
    # make_fx/torch.export may keep the dict structure or flatten it; the ABI
    # order is positional either way.
    if isinstance(got, dict):
        pairs = [(got[k], out[k]) for k in ("energy", "force", "virial")]
    else:
        flat = tuple(got) if isinstance(got, (tuple, list)) else (got,)
        assert len(flat) == 3, f"expected 3 outputs, got {len(flat)}"
        pairs = list(zip(flat, (out["energy"], out["force"], out["virial"])))
    for a, b in pairs:
        assert torch.allclose(a, b, atol=1e-10, rtol=0), f"max diff {(a-b).abs().max()}"


def test_energy_only_and_response_contract(model):
    eo_inputs = model.energy_only_sample_inputs()
    eo = model.trace_energy_only(*eo_inputs)
    assert output_keys_of(eo, eo_inputs) == ("energy", "latent_charge")
    with torch.no_grad():
        eo_out = eo(*eo_inputs)
    assert eo_out["energy"].shape == (1,)
    assert eo_out["latent_charge"].shape == (8, 1)

    r_inputs = model.response_sample_inputs()
    r = model.trace_response(*r_inputs)
    assert output_keys_of(r, r_inputs) == (
        "latent_charge",
        "charge_response",
        "charge_response_virial",
    )
    with torch.no_grad():
        out = r(*r_inputs)
    coord = r_inputs[10]
    v = r_inputs[12].reshape(-1)
    f_expected = -v.unsqueeze(-1) * model.w
    assert torch.allclose(out["charge_response"], f_expected, atol=1e-12, rtol=0)
    w_expected = f_expected.transpose(0, 1) @ coord
    assert torch.allclose(
        out["charge_response_virial"].reshape(3, 3), w_expected, atol=1e-12, rtol=0
    )


def test_install_artifacts_and_metadata(tmp_path):
    pt2 = tmp_path / "model.pt2"
    with zipfile.ZipFile(pt2, "w") as zf:
        zf.writestr("model/data.pkl", b"payload")

    meta = build_metadata(
        {
            "full_lr": ("energy", "force", "virial"),
            "energy_only": ("energy", "latent_charge"),
            "response_combined": ("energy", "force", "virial"),
        }
    )
    merged = install_artifacts(
        pt2,
        {"full_lr": b"FLR", "energy_only": b"EO", "response_combined": b"RC"},
        metadata=meta,
    )
    assert merged["has_full_lr_artifact"] is True
    assert merged["has_fused_response_artifacts"] is True
    assert merged["full_lr_output_keys"] == ["energy", "force", "virial"]

    with zipfile.ZipFile(pt2) as zf:
        names = zf.namelist()
    assert len(names) == len(set(names)), "duplicate zip entries"
    for kind in ("full_lr", "energy_only", "response_combined"):
        assert PT2_EXTRA_PREFIX + ARTIFACT_NAMES[kind] in names
    assert read_metadata(pt2)["has_full_lr_artifact"] is True

    # Re-installing must not duplicate entries (atomic rewrite).
    install_artifacts(pt2, {"full_lr": b"FLR2"})
    with zipfile.ZipFile(pt2) as zf:
        names = zf.namelist()
        assert zf.read(PT2_EXTRA_PREFIX + ARTIFACT_NAMES["full_lr"]) == b"FLR2"
    assert len(names) == len(set(names))


@pytest.mark.aoti
@pytest.mark.slow
def test_aoti_smoke(model, tmp_path):
    problems = check_toolchain()
    if problems:
        pytest.skip("; ".join(problems))
    inputs = model.full_lr_sample_inputs()
    traced = model.trace_full_lr(*inputs)
    with torch.no_grad():
        ref = traced(*inputs)
    exported = export_program(traced, inputs)
    try:
        payload = compile_artifact(
            exported, tmp_path / "full_lr.pt2", sample_inputs=inputs, cuda=False,
            source_fn=model.full_lr_callable,
        )
    except Exception as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"AOTI compilation unavailable: {type(exc).__name__}: {exc}")
    assert payload
    fn = load_package(tmp_path / "full_lr.pt2", device="cpu")
    with torch.no_grad():
        got = fn(*inputs)
    # The loaded callable may return the dict (torch >= 2.6 packages preserve
    # the pytree) or a flat tuple; normalise both.
    if isinstance(got, (tuple, list)) and len(got) == 1 and isinstance(got[0], dict):
        got = got[0]
    if isinstance(got, dict):
        pairs = [(got[k], ref[k]) for k in ("energy", "force", "virial")]
    else:
        flat = tuple(got) if isinstance(got, (tuple, list)) else (got,)
        assert len(flat) == 3, f"expected 3 outputs, got {len(flat)}"
        pairs = list(zip(flat, (ref["energy"], ref["force"], ref["virial"])))
    for a, b in pairs:
        assert torch.allclose(a, b, atol=1e-8, rtol=0), f"max diff {(a-b).abs().max()}"
