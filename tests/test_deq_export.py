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
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
)


def _build_memory(patch, *, extraction_block_class=None, rounds=3, basis="Z"):
    exp = MemoryExperiment(
        qec_patch=patch,
        extraction_block_class=extraction_block_class,
        rounds=rounds,
        noise_params=None,  # noiseless, per the Light-DEQ export convention
        basis=basis,
    )
    circuit = exp.build()
    return exp.system, circuit


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
        assert "OUTPUT memory " in deq_text

    def test_validates_structurally(self, deq_text):
        parsed = validate_deq_text(deq_text)
        if deq_available():
            from deq.circuit.model import CodeDefinition
            codes = [d for d in parsed.definitions if isinstance(d, CodeDefinition)]
            assert len(codes) == 1
            assert codes[0].n == 3 and codes[0].k == 1 and codes[0].d == 3
            assert len(codes[0].logicals) == 1
            assert len(codes[0].stabilizers) == 2


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


# --- multi-patch scope guard -------------------------------------------------

@pytest.mark.smoke
def test_multi_patch_raises_not_implemented():
    import stim

    system = QECSystem()
    system.add_patch(RepetitionCode(distance=3), name="a")
    system.add_patch(RepetitionCode(distance=3), name="b", offset=(0, 10))
    with pytest.raises(DeqExportError):
        export_deq(system, stim.Circuit(), gadget_name="X")
