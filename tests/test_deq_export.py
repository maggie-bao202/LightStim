"""
DEQ Export Tests — lightstim.deq.export_deq() structural validation.

Verifies exported .deq text is structurally correct and, when the real
`deq`/`deqagram` grammar packages are installed, round-trips through the
actual qdk-ec parser (not just a hand-rolled regex check).

Run:  pytest tests/test_deq_export.py -m smoke -q
"""
import numpy as np
import pytest

from lightstim.deq import DeqExportError, export_deq, pair_logical_operators
from lightstim.deq.validate import deq_available, validate_deq_text
from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.repetition.repetition import RepetitionCode
from lightstim.qec_code.repetition.SE_block import RepetitionCodeExtractionBlock
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
)


def _build_memory(patch, *, extraction_block_class=None, rounds=3, basis="Z", noise_params=None):
    exp = MemoryExperiment(
        qec_patch=patch,
        extraction_block_class=extraction_block_class,
        rounds=rounds,
        noise_params=noise_params,  # noiseless by default; noisy also exports fine (see below)
        basis=basis,
    )
    circuit = exp.build()
    return exp.system, circuit


def _build_prepare_and_se_only(patch, extraction_block_class, *, rounds=1):
    """A circuit that prepares data qubits and runs syndrome extraction but
    does NOT measure the data qubits out -- the code instance survives to
    the end, unlike every MemoryExperiment (which always ends in a full
    destructive readout). Built directly via the low-level IR (no
    protocol class does this shape today) so export_deq's OUTPUT-emission
    logic has a real positive case to be tested against."""
    from lightstim.ir.builder import CircuitBuilder
    from lightstim.ir.tracker import SyndromeTracker

    system = QECSystem()
    system.add_patch(patch, name="memory")
    tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker, system, if_detector=True)
    builder.write_coordinates()
    data = sorted(system.data_indices)
    builder.initialize({q: "Z" for q in data}, n=system.num_qubits)
    builder.apply_syndrome_extraction(
        circuit_chunk=extraction_block_class(system).circuit, rounds=rounds
    )
    return system, builder.circuit


def _build_two_patch_prepare_and_se_only(patch_class, extraction_block_class, *, rounds=1):
    """Two independent patches (like CNOTTransExperiment's control/target,
    minus the CNOT and the final readout), so both patches' qubits survive
    to the end and export_deq must emit two distinct, correctly-globalized
    OUTPUT lines."""
    from lightstim.ir.builder import CircuitBuilder
    from lightstim.ir.tracker import SyndromeTracker

    system = QECSystem()
    system.add_patch(patch_class(distance=3), name="control")
    system.add_patch(patch_class(distance=3), name="target", offset=(6, 0))
    tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker, system, if_detector=True)
    builder.write_coordinates()
    data = sorted(system.data_indices)
    builder.initialize({q: "Z" for q in data}, n=system.num_qubits)
    builder.apply_syndrome_extraction(
        circuit_chunk=extraction_block_class(system).circuit, rounds=rounds
    )
    return system, builder.circuit


# --- pairing.py: symplectic logical-operator pairing -----------------------

@pytest.mark.smoke
class TestPairLogicalOperators:

    def test_single_logical_qubit(self):
        data = [0, 1, 2]
        x_rec = {"pauli": {0: "X", 1: "X", 2: "X"}, "type": "X"}
        z_rec = {"pauli": {0: "Z"}, "type": "Z"}
        pairs = pair_logical_operators([z_rec, x_rec], data)  # insertion order Z,X
        assert pairs == [(x_rec, z_rec)]

    def test_adversarial_all_z_then_all_x_ordering(self):
        # Mirrors BB_code's _build_logical_operators_numerical path: all
        # Z records inserted first, then all X records — not alternating,
        # not even grouped by logical qubit, and the two Z's are listed in
        # the OPPOSITE order from their X partners. An insertion-order (or
        # "just zip x_ops with z_ops") heuristic would mispair this; the
        # symplectic approach must not.
        data = [0, 1]
        x0 = {"pauli": {0: "X"}, "type": "X"}  # logical qubit 0
        x1 = {"pauli": {1: "X"}, "type": "X"}  # logical qubit 1
        z0 = {"pauli": {0: "Z"}, "type": "Z"}  # logical qubit 0's partner
        z1 = {"pauli": {1: "Z"}, "type": "Z"}  # logical qubit 1's partner
        # all-Z-then-all-X, with z1 (partner of x1) listed BEFORE z0 (partner of x0)
        logical_ops = [z1, z0, x0, x1]
        pairs = pair_logical_operators(logical_ops, data)
        assert pairs == [(x0, z0), (x1, z1)]
        for x_rec, z_rec in pairs:
            x_qubits = set(x_rec["pauli"])
            z_qubits = set(z_rec["pauli"])
            assert len(x_qubits & z_qubits) % 2 == 1  # odd overlap => anticommute

    def test_mismatched_counts_raises(self):
        x_rec = {"pauli": {0: "X"}, "type": "X"}
        with pytest.raises(DeqExportError):
            pair_logical_operators([x_rec], [0, 1])

    def test_non_dual_basis_raises(self):
        # Two X's and two Z's that don't form a clean 1-1 anticommuting pairing
        # (both Z's anticommute with both X's => not a permutation matrix).
        data = [0, 1]
        x0 = {"pauli": {0: "X"}, "type": "X"}
        x1 = {"pauli": {1: "X"}, "type": "X"}
        z0 = {"pauli": {0: "Z", 1: "Z"}, "type": "Z"}
        z1 = {"pauli": {0: "Z"}, "type": "Z"}
        with pytest.raises(DeqExportError):
            pair_logical_operators([x0, x1, z0, z1], data)

    def test_no_logicals(self):
        assert pair_logical_operators([], [0, 1]) == []


# --- export_deq: repetition code (exact structural check) ------------------

@pytest.mark.smoke
class TestExportRepetitionCode:

    @staticmethod
    @pytest.fixture(scope="class")
    def deq_text():
        system, circuit = _build_memory(RepetitionCode(distance=3), rounds=3)
        return export_deq(system, circuit, gadget_name="Repetition")

    def test_code_header(self, deq_text):
        assert "CODE memory [[3,1,3]] {" in deq_text

    def test_one_logical_line(self, deq_text):
        assert deq_text.count("LOGICAL ") == 1

    def test_distance_minus_one_stabilizer_lines(self, deq_text):
        assert deq_text.count("STABILIZER ") == 2  # d=3 repetition code: 2 Z-checks

    def test_single_gadget(self, deq_text):
        assert deq_text.count("GADGET ") == 1

    def test_no_output_after_terminal_readout(self, deq_text):
        # MemoryExperiment ends in a full destructive data-qubit
        # measurement -- no live code instance survives, so OUTPUT must be
        # omitted (deq's compiler rejects an OUTPUT it can't verify against
        # actual measurements; see _destructively_measured_qubits).
        assert "OUTPUT" not in deq_text

    def test_validates_structurally(self, deq_text):
        parsed = validate_deq_text(deq_text)
        if deq_available():
            from deq.circuit.model import CodeDefinition
            codes = [d for d in parsed.definitions if isinstance(d, CodeDefinition)]
            assert len(codes) == 1
            assert codes[0].n == 3 and codes[0].k == 1 and codes[0].d == 3
            assert len(codes[0].logicals) == 1
            assert len(codes[0].stabilizers) == 2


@pytest.mark.smoke
class TestOutputEmission:
    """`OUTPUT` must be emitted exactly when the patch's qubits actually
    survive to the end of the GADGET body, and omitted when they don't
    (see `_destructively_measured_qubits`). `deq`'s own compiler enforces
    this: it rejects a GADGET whose OUTPUT stabilizers can't be
    reconstructed from the gadget's measurements, which is exactly what an
    OUTPUT over destructively-measured qubits claims. Caught for real by
    running the exported .deq for an H6-distillation circuit through
    Bloqade Studio's deq compiler (grammar-only `deqagram.parse` validation
    can't catch this class of error — it isn't a syntax problem)."""

    def test_output_present_when_code_survives(self):
        system, circuit = _build_prepare_and_se_only(
            RepetitionCode(distance=3), RepetitionCodeExtractionBlock, rounds=1
        )
        deq_text = export_deq(system, circuit, gadget_name="PrepareAndExtract")
        assert "OUTPUT memory " in deq_text
        validate_deq_text(deq_text)

    def test_output_absent_after_terminal_readout(self):
        system, circuit = _build_memory(RepetitionCode(distance=3), rounds=1)
        deq_text = export_deq(system, circuit, gadget_name="Repetition")
        assert "OUTPUT" not in deq_text
        validate_deq_text(deq_text)


@pytest.mark.smoke
def test_noisy_circuit_exports_and_validates():
    """Noisy circuits export fine: Stim's noise channels (X_ERROR,
    DEPOLARIZE1/2, ...) aren't reserved GADGET keywords, so they pass
    through as ordinary instructions. Do not confuse this with deq's own
    ERROR(p) <target> statement (CHECK/READOUT/LOGICAL only) — this module
    never emits that."""
    from lightstim.noise.config import NoiseConfig

    noise = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3)
    system, circuit = _build_memory(RepetitionCode(distance=3), rounds=2, noise_params=noise)
    assert "X_ERROR" in str(circuit) or "DEPOLARIZE" in str(circuit)  # sanity: actually noisy

    deq_text = export_deq(system, circuit, gadget_name="NoisyRepetition")
    assert "X_ERROR(" in deq_text or "DEPOLARIZE" in deq_text

    parsed = validate_deq_text(deq_text)
    if deq_available():
        from deq.circuit.model import Instruction
        gadget_instructions = {
            i.name
            for d in parsed.definitions
            for i in getattr(d, "body", [])
            if isinstance(i, Instruction)
        }
        assert gadget_instructions & {"X_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"}


# --- export_deq: rotated surface code (primary Light-DEQ target) -----------

@pytest.mark.smoke
class TestExportRotatedSurfaceCode:
    """Structural comparison against resource-superstaq's reference
    generated/rotated_surface_code_d3.deq (CODE RotatedSurfaceCodeW3H3
    [[9,1,3]], 8 stabilizers, 1 logical pair) — see the Light-DEQ plan."""

    @staticmethod
    @pytest.fixture(scope="class")
    def deq_text():
        system, circuit = _build_memory(
            RotatedSurfaceCode(distance=3),
            extraction_block_class=RotatedSurfaceCodeExtractionBlock,
            rounds=3,
        )
        return export_deq(system, circuit, gadget_name="RotatedSurfaceMemory")

    def test_code_matches_reference_parameters(self, deq_text):
        assert "CODE memory [[9,1,3]] {" in deq_text

    def test_eight_stabilizers(self, deq_text):
        assert deq_text.count("STABILIZER ") == 8

    def test_one_logical_pair(self, deq_text):
        assert deq_text.count("LOGICAL ") == 1

    def test_validates_structurally(self, deq_text):
        parsed = validate_deq_text(deq_text)
        if deq_available():
            from deq.circuit.model import CodeDefinition, GadgetDefinition
            codes = [d for d in parsed.definitions if isinstance(d, CodeDefinition)]
            gadgets = [d for d in parsed.definitions if isinstance(d, GadgetDefinition)]
            assert len(codes) == 1
            assert (codes[0].n, codes[0].k, codes[0].d) == (9, 1, 3)
            assert len(codes[0].stabilizers) == 8
            assert len(codes[0].logicals) == 1
            assert len(gadgets) == 1


# --- multi-observable ordering convention (k=2 synthetic probe) ------------

@pytest.mark.smoke
@pytest.mark.skipif(not deq_available(), reason="requires deq/deqagram")
def test_readout_statement_order_matches_logical_declaration_order():
    """The Nth READOUT/OBSERVABLE_INCLUDE in a GADGET body corresponds, by
    position, to the Nth LOGICAL declared in the associated CODE block.
    Confirmed empirically against the real grammar (no k>1 example ships
    with qdk-ec) — see Light-DEQ plan §3c."""
    from deq.circuit import parser
    text = """
CODE TwoQubitCode [[4,2]] {
    LOGICAL X0*X1 Z0*Z2
    LOGICAL X2*X3 Z1*Z3
    STABILIZER Z0*Z1*Z2*Z3
}

GADGET Test {
    R 0 1 2 3
    M 0 1 2 3
    OBSERVABLE_INCLUDE rec[-4] rec[-3]
    OBSERVABLE_INCLUDE rec[-2] rec[-1]
}
"""
    f = parser.parse(text)
    from deq.circuit.model import GadgetDefinition, ReadoutStatement
    gadget = next(d for d in f.definitions if isinstance(d, GadgetDefinition))
    readouts = [s for s in gadget.body if isinstance(s, ReadoutStatement)]
    assert len(readouts) == 2  # one per LOGICAL, in declaration order


# --- multi-patch export -----------------------------------------------------

@pytest.mark.smoke
def test_empty_system_raises():
    import stim

    system = QECSystem()
    with pytest.raises(DeqExportError):
        export_deq(system, stim.Circuit(), gadget_name="X")


@pytest.mark.smoke
def test_duplicate_code_name_raises():
    import stim

    system = QECSystem()
    system.add_patch(RepetitionCode(distance=3), name="a")
    system.add_patch(RepetitionCode(distance=3), name="b", offset=(0, 10))
    with pytest.raises(DeqExportError):
        export_deq(
            system,
            stim.Circuit(),
            gadget_name="X",
            code_names={"a": "same", "b": "same"},
        )


@pytest.mark.smoke
class TestExportTwoPersistingPatches:
    """Two independent, persisting patches (no coupler) — CNOTTransExperiment."""

    @staticmethod
    @pytest.fixture(scope="class")
    def deq_text():
        from lightstim.protocols.cnot_trans import CNOTTransExperiment

        exp = CNOTTransExperiment(
            code_patch_class=RotatedSurfaceCode,
            extraction_block_class=RotatedSurfaceCodeExtractionBlock,
            code_params_control={"distance": 3},
            rounds_before=1,
            rounds_after=1,
            noise_params=None,
        )
        circuit = exp.build()
        return export_deq(exp.system, circuit, gadget_name="CNOTTrans")

    def test_two_code_blocks(self, deq_text):
        assert deq_text.count("CODE ") == 2
        assert "CODE control [[9,1,3]] {" in deq_text
        assert "CODE target [[9,1,3]] {" in deq_text

    def test_single_gadget(self, deq_text):
        assert deq_text.count("GADGET ") == 1

    def test_no_output_after_terminal_readout(self, deq_text):
        # CNOTTransExperiment.build() ends in a full destructive data-qubit
        # readout for both patches -- no live code instance survives either,
        # so no OUTPUT is emitted (see TestOutputEmission). The distinct-
        # global-qubit regression is covered separately by
        # test_two_patches_get_distinct_global_qubits, on a circuit that
        # doesn't end in readout so OUTPUT is actually present to check.
        assert "OUTPUT" not in deq_text

    def test_validates_structurally(self, deq_text):
        parsed = validate_deq_text(deq_text)
        if deq_available():
            from deq.circuit.model import CodeDefinition
            codes = {d.name: d for d in parsed.definitions if isinstance(d, CodeDefinition)}
            assert set(codes) == {"control", "target"}
            for c in codes.values():
                assert (c.n, c.k, c.d) == (9, 1, 3)
                assert len(c.stabilizers) == 8


@pytest.mark.smoke
def test_coupler_patch_exported_as_stabilizer_state():
    """Lattice-surgery coupler patches get their own CODE block too (a k=0
    stabilizer-state code, same as the resource-superstaq reference file's
    YBoundaryStateD3 [[8,0]]), whether or not the protocol has removed the
    coupler from system.patches by the time .build() returns. (No OUTPUT
    assertions here: TwoPatchLSExperiment ends in a full destructive
    readout of every patch including the coupler, so none of them get an
    OUTPUT -- see TestOutputEmission for that behavior specifically.)"""
    import contextlib
    import io

    from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment

    with contextlib.redirect_stdout(io.StringIO()):
        exp = TwoPatchLSExperiment(
            patch1_config={"distance": 3},
            patch2_config={"distance": 3},
            offset=(0, 10),
            interaction_type="ZZ",
            initial_state_patch1="X",
            initial_state_patch2="Z",
            measure_state_patch1="X",
            measure_state_patch2="Z",
            rounds=1,
            noise_params=None,
        )
        circuit = exp.build()

    assert exp.system.coupler_patches, "test assumes the coupler is still present after build()"
    coupler_name = next(iter(exp.system.coupler_patches))
    deq_text = export_deq(exp.system, circuit, gadget_name="TwoPatchLS")

    assert deq_text.count("CODE ") == 3  # two code patches + the coupler
    assert f"CODE {coupler_name} " in deq_text
    assert "LOGICAL" not in deq_text.split(f"CODE {coupler_name} ", 1)[1].split("}", 1)[0]

    parsed = validate_deq_text(deq_text)
    if deq_available():
        from deq.circuit.model import CodeDefinition
        codes = {d.name: d for d in parsed.definitions if isinstance(d, CodeDefinition)}
        assert set(codes) == set(exp.system.patches)
        coupler_code = codes[coupler_name]
        assert coupler_code.k == 0
        assert len(coupler_code.logicals) == 0
        assert len(coupler_code.stabilizers) > 0


@pytest.mark.smoke
def test_two_patches_get_distinct_global_qubits():
    """Regression test: a non-first patch's OUTPUT must use its own physical
    (global) qubits, not silently alias the first patch's local-index-shaped
    numbers. (A prior version of export_deq read patch.data_indices/
    patch.stabilizers directly, which are in LightStim's per-patch LOCAL
    index space; for any patch after the first, local != global, and the
    exported OUTPUT pointed at the wrong qubits.)

    Uses a prepare+SE-only circuit (see _build_two_patch_prepare_and_se_only)
    rather than the full CNOTTransExperiment, specifically so both patches'
    qubits survive to the end and OUTPUT is actually emitted for both --
    CNOTTransExperiment's real .build() ends in a full destructive readout,
    which (correctly, per TestOutputEmission) suppresses OUTPUT entirely and
    would give this regression nothing to check.
    """
    system, circuit = _build_two_patch_prepare_and_se_only(
        RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock, rounds=1
    )
    deq_text = export_deq(system, circuit, gadget_name="TwoPatchPrepareAndExtract")

    control_local_to_global = system.local_to_global_map["control"]
    target_local_to_global = system.local_to_global_map["target"]
    assert control_local_to_global != target_local_to_global  # sanity: truly different patches

    # patch.data_indices is LightStim's own LOCAL numbering (confirmed
    # reliable for ordinary, non-coupler patches; see _globalize_pauli_map).
    control_local_data = system.patches["control"][0].data_indices
    target_local_data = system.patches["target"][0].data_indices
    expected_control = sorted(control_local_to_global[q] for q in control_local_data)
    expected_target = sorted(target_local_to_global[q] for q in target_local_data)
    assert expected_control != expected_target

    control_output = " ".join(map(str, expected_control))
    target_output = " ".join(map(str, expected_target))
    assert f"OUTPUT control {control_output}" in deq_text
    assert f"OUTPUT target {target_output}" in deq_text

    parsed = validate_deq_text(deq_text)
    if deq_available():
        from deq.circuit.model import CodeDefinition
        codes = [d for d in parsed.definitions if isinstance(d, CodeDefinition)]
        assert len(codes) == 2
