"""Export LightStim circuits to Microsoft's ``.deq`` DSL.

Converts a LightStim ``QECSystem`` (for its ``CODE`` block: stabilizers and
logical operators) plus the matching NOISELESS ``stim.Circuit`` (for the
``GADGET`` body/bodies) into ``.deq`` text.

Scope (v1): single-patch systems only (the common ``MemoryExperiment``
case). Multi-patch export (lattice surgery, code deformation) is future
work — see the Light-DEQ plan.

The exported circuit is expected to be noiseless. DEQ's ``ERROR(p) <target>``
statement only accepts CHECK/READOUT/LOGICAL targets, not per-qubit Pauli
targets, so it is not a drop-in replacement for Stim's ``X_ERROR(p) q...``.
Matching the convention already established by the (now-deleted)
``surface_code_deq`` generator this replaces, LightStim should export the
ideal circuit and treat noise injection as a separate, later concern.
"""

import stim

from lightstim.ir.qec_system import QECSystem

from .pairing import DeqExportError, pair_logical_operators

__all__ = ["DeqExportError", "export_deq"]


# Pure LightStim geometry, meaningless to deq consumers.
_DROP_INSTRUCTIONS = {"QUBIT_COORDS", "SHIFT_COORDS"}
# Native deq synonyms of CHECK/READOUT that carry no (...) argument slot in
# the deq grammar, unlike Stim's own DETECTOR(x, y, t) / OBSERVABLE_INCLUDE(k).
_STRIP_ARGS_INSTRUCTIONS = {"DETECTOR", "OBSERVABLE_INCLUDE"}


def _format_float(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def _target_text(target: "stim.GateTarget") -> str:
    prefix = "!" if target.is_inverted_result_target else ""
    if target.is_combiner:
        return "*"
    if target.is_measurement_record_target:
        return f"rec[{target.value}]"
    if target.is_x_target:
        return f"{prefix}X{target.value}"
    if target.is_y_target:
        return f"{prefix}Y{target.value}"
    if target.is_z_target:
        return f"{prefix}Z{target.value}"
    if target.is_sweep_bit_target:
        return f"sweep[{target.value}]"
    if target.is_qubit_target:
        return f"{prefix}{target.value}"
    raise DeqExportError(f"unsupported stim target for .deq export: {target!r}")


def _tag_text(tag: str) -> str:
    escaped = tag.replace("\\", "\\B").replace("]", "\\C").replace("\r", "\\r").replace("\n", "\\n")
    return f"[{escaped}]"


def _instruction_line(instruction: "stim.CircuitInstruction") -> str | None:
    name = instruction.name
    if name in _DROP_INSTRUCTIONS:
        return None
    tag_text = _tag_text(instruction.tag) if instruction.tag else ""
    targets_text = " ".join(_target_text(t) for t in instruction.targets_copy())
    if name in _STRIP_ARGS_INSTRUCTIONS:
        return f"{name}{tag_text} {targets_text}".rstrip()
    args = instruction.gate_args_copy()
    args_text = f"({', '.join(_format_float(a) for a in args)})" if args else ""
    return f"{name}{tag_text}{args_text} {targets_text}".rstrip()


def _circuit_lines(circuit: "stim.Circuit", indent: int = 1) -> list[str]:
    pad = "    " * indent
    lines: list[str] = []
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            lines.append(f"{pad}REPEAT {instruction.repeat_count} {{")
            lines.extend(_circuit_lines(instruction.body_copy(), indent=indent + 1))
            lines.append(f"{pad}}}")
        else:
            line = _instruction_line(instruction)
            if line is not None:
                lines.append(f"{pad}{line}")
    return lines


def _pauli_product_text(pauli_map: dict, local_id: dict) -> str:
    terms = sorted(pauli_map.items(), key=lambda kv: local_id[kv[0]])
    return "*".join(f"{pauli}{local_id[qubit]}" for qubit, pauli in terms)


def _infer_distance(patch):
    """Best-effort code distance lookup across LightStim's varied attribute names."""
    if (d := getattr(patch, "distance", None)) is not None:
        return d
    dz, dx = getattr(patch, "distance_z", None), getattr(patch, "distance_x", None)
    if dz is not None and dz == dx:
        return dz
    return None


def _code_block_text(patch, code_name: str) -> str:
    data_indices = sorted(patch.data_indices)
    local_id = {qubit: i for i, qubit in enumerate(data_indices)}
    n = len(data_indices)
    k = patch.num_logicals
    distance = _infer_distance(patch)
    params = f"[[{n},{k},{distance}]]" if distance is not None else f"[[{n},{k}]]"

    lines = [f"CODE {code_name} {params} {{"]
    if k > 0:
        pairs = pair_logical_operators(patch.logical_ops, data_indices)
        if len(pairs) != k:
            raise DeqExportError(
                f"patch {code_name!r} declares num_logicals={k} but "
                f"{len(pairs)} X/Z logical pairs were recovered from logical_ops."
            )
        for x_rec, z_rec in pairs:
            x_text = _pauli_product_text(x_rec["pauli"], local_id)
            z_text = _pauli_product_text(z_rec["pauli"], local_id)
            lines.append(f"    LOGICAL {x_text} {z_text}")
    if not patch.stabilizers:
        raise DeqExportError(f"patch {code_name!r} has no stabilizers to export")
    for stabilizer in patch.stabilizers:
        lines.append(f"    STABILIZER {_pauli_product_text(stabilizer['pauli'], local_id)}")
    lines.append("}")
    return "\n".join(lines)


def _gadget_block_text(
    name: str,
    body: "stim.Circuit",
    *,
    input_port: tuple | None = None,
    output_port: tuple | None = None,
) -> str:
    lines = [f"GADGET {name} {{"]
    if input_port is not None:
        code_name, qubits = input_port
        lines.append(f"    INPUT {code_name} {' '.join(str(q) for q in qubits)}")
    lines.extend(_circuit_lines(body, indent=1))
    if output_port is not None:
        code_name, qubits = output_port
        lines.append(f"    OUTPUT {code_name} {' '.join(str(q) for q in qubits)}")
    lines.append("}")
    return "\n".join(lines)


def export_deq(
    system: QECSystem,
    circuit: "stim.Circuit",
    *,
    gadget_name: str = "Circuit",
    code_names: dict | None = None,
) -> str:
    """Export a single-patch LightStim circuit to ``.deq`` text.

    Args:
        system: A ``QECSystem`` with exactly one patch (the common
            ``MemoryExperiment(qec_patch=...)`` case). Multi-patch export
            is not yet supported.
        circuit: The NOISELESS ``stim.Circuit`` for this system (e.g.
            ``MemoryExperiment(..., noise_params=None).build()``).
        gadget_name: Name for the single emitted GADGET.
        code_names: Optional override mapping patch name -> CODE identifier.
            Defaults to the patch name itself.

    Returns:
        The full ``.deq`` file text.

    Note:
        v1 emits the whole circuit as ONE monolithic GADGET, deliberately
        *not* split into separate Prepare/SyndromeExtraction/Measure gadgets
        wired via COMPOSE (unlike the resource-superstaq reference file this
        replaces). Splitting naively is unsound: deq scopes ``rec[-k]``
        *per GADGET*, so a DETECTOR in a later gadget that compares against a
        measurement from an earlier gadget (the normal case for consecutive
        QEC rounds) would be out of range once split. Crossing a gadget
        boundary correctly requires deq's ``IN<p>.S<s>`` virtual-port syntax,
        which needs a detector -> stabilizer-index mapping LightStim doesn't
        currently expose — real design work, tracked as follow-up. Keeping
        everything in one GADGET (with Stim's native REPEAT block preserved
        as-is) sidesteps the issue entirely: rec[-k] stays contiguous.
    """
    if len(system.patches) != 1:
        raise DeqExportError(
            f"export_deq only supports single-patch systems for now; got "
            f"{len(system.patches)} patches ({sorted(system.patches)}). "
            "Multi-patch export is future work."
        )
    (patch_name, (patch, _offset)) = next(iter(system.patches.items()))
    code_name = (code_names or {}).get(patch_name, patch_name)
    output_qubits = sorted(patch.data_indices)

    sections = [
        _code_block_text(patch, code_name),
        _gadget_block_text(gadget_name, circuit, output_port=(code_name, output_qubits)),
    ]

    header = "# Generated by lightstim.deq.export; do not edit by hand."
    return header + "\n\n" + "\n\n".join(sections) + "\n"
