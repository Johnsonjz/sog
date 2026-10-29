"""The two artifact families and their metadata contract.

Both deployment chains are supported:

**Mode A — pair + kspace + fix (2F+1B, the compatibility default)**

* ``sog_energy_only.pt2`` — the pair's forward: returns ``(energy, latent_charge)``
  and no force; the LAMMPS C++ kspace solver consumes the charges.
* ``latent_charge_response.pt2`` / ``_with_comm.pt2`` — the response VJP: takes
  the graph plus ``v_charge`` and returns ``charge_response`` and
  ``charge_response_virial``; ``fix sog/response`` adds them.
* ``sog_response_combined.pt2`` — the fused variant: seed = ``E_SR + sum(v q)``,
  one backward yields ``F_SR + F_resp`` (still pair+kspace+fix; 2F+1B).

**Mode B — full-LR single graph (1F+1B)**

* ``sog_full_lr.pt2`` (12 inputs) / ``sog_full_lr_wc.pt2`` (20 inputs) —
  SR + LR in one graph with the in-graph ``sogkspace`` solve; outputs
  ``energy``, ``force``, ``virial``.

Metadata flag names match the DeepMD freeze exactly, so existing consumers
keep working.  All entries are written under ``model/extra/`` inside the
``.pt2`` zip.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any, Mapping

PT2_EXTRA_PREFIX = "model/extra/"

ARTIFACT_NAMES = {
    "energy_only": "sog_energy_only.pt2",
    "response": "latent_charge_response.pt2",
    "response_wc": "latent_charge_response_with_comm.pt2",
    "response_combined": "sog_response_combined.pt2",
    "full_lr": "sog_full_lr.pt2",
    "full_lr_wc": "sog_full_lr_wc.pt2",
}

#: Metadata flags written when the corresponding artifact is present.
#: ``energy_only`` + ``response_combined`` jointly set
#: ``has_fused_response_artifacts`` (both are needed for the fused Mode-A chain).
ARTIFACT_FLAGS = {
    "response": "has_charge_response_artifact",
    "response_wc": "has_charge_response_with_comm_artifact",
    "full_lr": "has_full_lr_artifact",
    "full_lr_wc": "has_full_lr_artifact_wc",
}

_OUTPUT_KEY_NAMES = {
    "energy_only": "fused_energy_only_output_keys",
    "response_combined": "fused_combined_output_keys",
    "full_lr": "full_lr_output_keys",
    "response": "charge_response_output_keys",
}

METADATA_ENTRY = PT2_EXTRA_PREFIX + "metadata.json"


def artifact_entry(kind_or_name: str) -> str:
    """Zip entry (``model/extra/<file>``) for an artifact kind or file name."""
    name = ARTIFACT_NAMES.get(kind_or_name, kind_or_name)
    return PT2_EXTRA_PREFIX + name


def build_metadata(
    present: Mapping[str, Any],
    *,
    base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge artifact presence flags/output keys into a metadata dict.

    ``present`` maps an artifact kind to either ``True`` or an iterable of
    output-key names (which also records ``*_output_keys`` where the DeepMD
    consumers expect it).
    """
    meta = dict(base or {})
    for kind, value in present.items():
        if kind not in ARTIFACT_FLAGS and kind not in _OUTPUT_KEY_NAMES:
            raise KeyError(f"unknown artifact kind {kind!r}")
        if kind in ARTIFACT_FLAGS:
            meta[ARTIFACT_FLAGS[kind]] = True
        if isinstance(value, (list, tuple)):
            meta[_OUTPUT_KEY_NAMES[kind]] = list(value)
    if "energy_only" in present and "response_combined" in present:
        meta["has_fused_response_artifacts"] = True
    return meta


def read_metadata(pt2_path: str | os.PathLike) -> dict[str, Any]:
    """Read ``model/extra/metadata.json`` from a ``.pt2`` archive (or ``{}``)."""
    with zipfile.ZipFile(pt2_path) as zf:
        names = set(zf.namelist())
        if METADATA_ENTRY not in names:
            return {}
        return json.loads(zf.read(METADATA_ENTRY).decode("utf-8"))


def install_artifacts(
    pt2_path: str | os.PathLike,
    artifacts: Mapping[str, bytes],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Add/replace SOG artifacts + metadata inside an existing ``.pt2`` zip.

    Rewrites the archive atomically (read all entries, merge, replace) so
    duplicate entries never accumulate.  Returns the merged metadata.
    """
    path = Path(pt2_path)
    with zipfile.ZipFile(path) as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}

    existing = (
        json.loads(entries[METADATA_ENTRY].decode("utf-8"))
        if METADATA_ENTRY in entries
        else {}
    )
    if metadata:
        existing.update(dict(metadata))
    entries[METADATA_ENTRY] = json.dumps(existing).encode("utf-8")

    for kind, payload in artifacts.items():
        entries[artifact_entry(kind)] = bytes(payload)

    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    os.replace(tmp, path)
    return existing
