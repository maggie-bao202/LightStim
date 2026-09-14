"""Export LightStim circuits to Microsoft's ``.deq`` DSL.

Converts a LightStim ``QECSystem`` (for its ``CODE`` block: stabilizers and
logical operators) plus the matching ``stim.Circuit`` (for the ``GADGET``
body) into ``.deq`` text.

Any number of patches is supported: one ``CODE`` block per patch, in a
single GADGET, each with an ``OUTPUT`` declaration IF AND ONLY IF that
patch's qubits still hold live quantum state at the end of the circuit (see
``_destructively_measured_qubits``) — a circuit ending in a full data-qubit
readout (every ``MemoryExperiment``, for instance) emits no ``OUTPUT`` at
all, matching the resource-superstaq reference file's own convention
(compare its ``PrepareZ``/``SyndromeExtraction`` gadgets, which end mid-code
and declare ``OUTPUT``, against ``MeasureZ``, which doesn't). This isn't
just a style choice: ``deq``'s own compiler verifies that every declared
``OUTPUT`` stabilizer is reconstructible from the gadget's measurements
("automatic check discovery"), and correctly rejects a GADGET that claims a
destructively-measured qubit set is still a valid code instance — caught for
real running an early, over-eager version of this exporter's output through
Bloqade Studio's ``deq`` compile step (grammar-only ``deqagram.parse``
validation, which this module still relies on day to day since there's no
local Rust toolchain, does NOT catch this class of error — it isn't a syntax
problem).

Decomposing a circuit into separate Prepare/SyndromeExtraction/Measure/merge
GADGETs wired via ``COMPOSE`` (as LightStim's lattice-surgery protocols
would naturally map onto) is still future work — see the Light-DEQ plan and
the note on ``export_deq`` for why that split isn't just a refactor.

Noisy circuits export fine: deq's ``instruction`` grammar rule is a generic
``IDENT tag? (args)? target*`` and Stim's noise channels (``X_ERROR``,
``DEPOLARIZE1``/``DEPOLARIZE2``, etc.) are not reserved GADGET keywords, so
they pass through as ordinary instructions — confirmed by round-tripping a
noisy ``MemoryExperiment`` circuit through the real ``deq``/``deqagram``
grammar. (Do not confuse this with deq's own native ``ERROR(p) <target>``
statement, which is a different thing: it only accepts CHECK/READOUT/LOGICAL
targets and injects a probabilistic flip on an already-abstracted outcome,
not per-qubit physical noise — this module never emits it.)
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
# Plain (non-resetting) measurements: after one of these, a qubit holds no
# further live quantum state (unlike MR/MRX/MRY, which reset it back to a
# known state it could still carry code data in). Used to detect a qubit
# that's been destructively measured out by the end of the GADGET body.
_TERMINAL_MEASUREMENT_INSTRUCTIONS = {"M", "MZ", "MX", "MY"}


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


def _destructively_measured_qubits(circuit: "stim.Circuit") -> set:
    """Qubits whose LAST touch in `circuit` is a plain (non-resetting)
    measurement — i.e. they hold no live quantum state by the end of the
    GADGET body, having been destructively measured out rather than
    handed off as a still-valid code instance.

    Used to decide whether a patch's OUTPUT declaration should be emitted
    at all: `deq`'s compiler verifies that every declared OUTPUT stabilizer
    is reconstructible from the gadget's own measurements (its "automatic
    check discovery"), and correctly rejects an OUTPUT over qubits that
    were actually consumed by a terminal readout — there's no code left to
    output. Matches the convention already used by the resource-superstaq
    reference file: its `MeasureZ`/`MeasureX` gadgets end in a plain `MZ`/
    `MX` and declare no `OUTPUT` at all.
    """
    last_is_measurement: dict = {}
    for instruction in circuit.flattened():
        is_measurement = instruction.name in _TERMINAL_MEASUREMENT_INSTRUCTIONS
        for target in instruction.targets_copy():
            if target.is_qubit_target:
                last_is_measurement[target.value] = is_measurement
    return {q for q, measured in last_is_measurement.items() if measured}


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


def _globalize_pauli_map(pauli_map: dict, patch_name: str, system: QECSystem) -> dict:
    """Convert one stabilizer/logical-operator Pauli map's keys to global
    circuit-qubit indices (i.e. the indices the physical ``stim.Circuit``
    actually uses).

    Two different per-patch representations exist in LightStim today, and
    neither is already in global-qubit terms:

    * An ordinary ``QECPatch`` (built via ``create_stim_stabilizer``/
      ``create_stim_logical``) keys its ``.stabilizers``/``.logical_ops``
      Pauli maps by the patch's own LOCAL qubit indices (0..n-1, as assigned
      by the patch's own ``.build()``) — including ``patch.data_indices``
      itself, despite it looking already-global for a first-added patch
      (local happens to equal global there only because
      ``QECSystem.local_to_global_map`` is the identity map for the first
      patch added). For any later-added patch local != global, and reading
      ``patch.data_indices``/pauli-map keys directly without converting
      through ``system.local_to_global_map[patch_name]`` silently produces
      the WRONG physical qubits (confirmed: a naive read had a two-patch
      export's second patch's OUTPUT accidentally aliasing the first
      patch's qubits).
    * A lattice-surgery *coupler* patch (``lightstim.ir.coupler.
      LogicalCouplerPatch``) keys its ``.stabilizers``/``.logical_ops``
      Pauli maps by raw LOCAL COORDINATE tuples instead of even local
      integers — a further, patch-kind-specific gap in its construction
      (not something this module can fix at the source).

    Both cases are handled here: an integer key is treated as a local
    LightStim index and converted via ``system.local_to_global_map``; a
    tuple key is treated as a coordinate and converted via
    ``system.index_map`` (coordinates in a coupler patch are already
    expressed in the system's global coordinate frame, since couplers are
    registered with ``offset=(0, 0)``).
    """
    if not pauli_map:
        return {}
    sample_key = next(iter(pauli_map))
    if isinstance(sample_key, tuple):
        return {system.index_map[coord]: pauli for coord, pauli in pauli_map.items()}
    local_to_global = system.local_to_global_map[patch_name]
    return {local_to_global[local_idx]: pauli for local_idx, pauli in pauli_map.items()}


def _code_qubit_universe(stabilizers: list, logical_ops: list) -> list:
    """The full set of (already-global) qubits a patch's CODE block must
    declare: every qubit touched by its stabilizers or logical operators."""
    qubits = set()
    for stabilizer in stabilizers:
        qubits.update(stabilizer["pauli"])
    for logical_op in logical_ops:
        qubits.update(logical_op["pauli"])
    return sorted(qubits)


def _code_block_text(
    patch, code_name: str, stabilizers: list, logical_ops: list, qubits: list
) -> str:
    """Render a CODE block for `patch` over the (global-index) qubit space
    `qubits`, using already-globalized `stabilizers`/`logical_ops` (see
    `_globalize_pauli_map`)."""
    local_id = {qubit: i for i, qubit in enumerate(qubits)}
    n = len(qubits)
    k = patch.num_logicals
    distance = _infer_distance(patch)
    params = f"[[{n},{k},{distance}]]" if distance is not None else f"[[{n},{k}]]"

    lines = [f"CODE {code_name} {params} {{"]
    if k > 0:
        pairs = pair_logical_operators(logical_ops, qubits)
        if len(pairs) != k:
            raise DeqExportError(
                f"patch {code_name!r} declares num_logicals={k} but "
                f"{len(pairs)} X/Z logical pairs were recovered from logical_ops."
            )
        for x_rec, z_rec in pairs:
            x_text = _pauli_product_text(x_rec["pauli"], local_id)
            z_text = _pauli_product_text(z_rec["pauli"], local_id)
            lines.append(f"    LOGICAL {x_text} {z_text}")
    if not stabilizers:
        raise DeqExportError(f"patch {code_name!r} has no stabilizers to export")
    for stabilizer in stabilizers:
        lines.append(f"    STABILIZER {_pauli_product_text(stabilizer['pauli'], local_id)}")
    lines.append("}")
    return "\n".join(lines)


def _gadget_block_text(
    name: str,
    body: "stim.Circuit",
    *,
    input_ports: list = (),
    output_ports: list = (),
) -> str:
    lines = [f"GADGET {name} {{"]
    for code_name, qubits in input_ports:
        lines.append(f"    INPUT {code_name} {' '.join(str(q) for q in qubits)}")
    lines.extend(_circuit_lines(body, indent=1))
    for code_name, qubits in output_ports:
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
    """Export a LightStim circuit to ``.deq`` text.

    Args:
        system: A ``QECSystem`` with one or more patches. Each patch becomes
            its own ``CODE`` block (with its own local qubit numbering) and
            its own ``OUTPUT`` declaration in the single emitted GADGET —
            e.g. a two-patch transversal-gate experiment
            (``CNOTTransExperiment``) exports one ``CODE``/``OUTPUT`` pair
            per patch, both declared in the same GADGET body.
        circuit: The ``stim.Circuit`` for this system, e.g.
            ``MemoryExperiment(...).build()`` — noisy or noiseless, both
            export fine (see the module docstring).
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

        Lattice-surgery-style protocols that register a transient *coupler*
        patch (``QECSystem.register_coupler``) export cleanly, including any
        that are still live in ``system.patches``/``system.coupler_patches``
        when ``export_deq`` runs (protocols vary on whether they remove the
        coupler by the end of ``.build()``): a coupler patch gets its own
        ``CODE`` block too, typically a stabilizer-state code (``k=0``, no
        ``LOGICAL`` line), the same way the resource-superstaq reference
        file's ``YBoundaryStateD3 [[8,0]]`` represents its own mediator
        patch. Every patch's CODE block is built from its OWN canonical,
        per-patch ``.stabilizers``/``.logical_ops`` (globalized on the fly —
        see ``_globalize_pauli_map``), not from ``system.stabilizers``/
        ``system.logical_ops``: the system-level lists track the
        Heisenberg-EVOLVED logical frame (needed for decoding across
        entangling gates like a transversal CNOT or a lattice-surgery
        merge), which is the wrong thing for a CODE declaration — a CODE is
        the code's fixed, intrinsic definition, independent of circuit
        history. Reading the per-patch attribute instead keeps a patch's
        CODE block stable regardless of what gates were later applied to it.
    """
    if not system.patches:
        raise DeqExportError("system has no patches to export")

    consumed_qubits = _destructively_measured_qubits(circuit)

    code_names = code_names or {}
    sections = []
    output_ports = []
    seen_code_names: dict = {}
    for patch_name, (patch, _offset) in system.patches.items():
        code_name = code_names.get(patch_name, patch_name)
        if code_name in seen_code_names:
            raise DeqExportError(
                f"patches {seen_code_names[code_name]!r} and {patch_name!r} both map to "
                f"CODE name {code_name!r}; pass `code_names` to disambiguate."
            )
        seen_code_names[code_name] = patch_name
        stabilizers = [
            {**s, "pauli": _globalize_pauli_map(s["pauli"], patch_name, system)}
            for s in patch.stabilizers
        ]
        logical_ops = [
            {**lo, "pauli": _globalize_pauli_map(lo["pauli"], patch_name, system)}
            for lo in patch.logical_ops
        ]
        qubits = _code_qubit_universe(stabilizers, logical_ops)
        sections.append(_code_block_text(patch, code_name, stabilizers, logical_ops, qubits))
        # Skip OUTPUT entirely if this patch's qubits were all destructively
        # measured out (a terminal readout, e.g. ending in MX) — there's no
        # live code instance left to claim as this CODE's OUTPUT, and
        # deq's compiler correctly rejects that claim (see
        # _destructively_measured_qubits' docstring).
        if not (set(qubits) <= consumed_qubits):
            output_ports.append((code_name, qubits))

    sections.append(_gadget_block_text(gadget_name, circuit, output_ports=output_ports))

    header = "# Generated by lightstim.deq.export; do not edit by hand."
    return header + "\n\n" + "\n\n".join(sections) + "\n"
