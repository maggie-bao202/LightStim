"""Pairing of X/Z logical-operator records into per-logical-qubit pairs.

DEQ's ``LOGICAL`` statement takes one X operator and one Z operator per
logical qubit (``LOGICAL X0*X1*X2 Z0*Z1*Z2``). LightStim's ``QECPatch``/
``QECSystem`` store ``logical_ops`` as a flat list of ``{"pauli": ..., "type":
"X"|"Z"}`` records with no explicit pairing between an X record and its
partner Z record, and insertion order is not a reliable pairing signal:
different code implementations insert X/Z records in different relative
orders (e.g. Z-then-X for the repetition code, X,Z,Z,X interleaved for one
BB_code construction path and all-Z-then-all-X for another). Pairing is
therefore recovered algebraically: X_i and Z_i on the same logical qubit
anticommute, while X_i and Z_j (i != j) commute, so the anticommutation
matrix between all X records and all Z records is a permutation matrix
identifying the correct pairing.
"""

import numpy as np

from lightstim.utils.linear_algebra import check_commutativity


class DeqExportError(Exception):
    """Raised when LightStim IR data cannot be faithfully exported to .deq."""


def _to_symplectic(record: dict, col: dict, n: int) -> np.ndarray:
    v = np.zeros(2 * n, dtype=np.uint8)
    for qubit, pauli in record["pauli"].items():
        i = col[qubit]
        if pauli in ("X", "Y"):
            v[i] = 1
        if pauli in ("Z", "Y"):
            v[n + i] = 1
    return v


def pair_logical_operators(logical_ops: list, data_indices) -> list:
    """Pair X-type and Z-type logical-operator records by logical qubit.

    Args:
        logical_ops: Flat list of records shaped like ``QECPatch.logical_ops``
            (``{"pauli": {qubit: 'X'|'Y'|'Z'}, "type": 'X'|'Z', ...}``).
        data_indices: The code's data-qubit indices (defines the symplectic
            vector space; order doesn't matter, only the set).

    Returns:
        A list of ``(x_record, z_record)`` tuples, one per logical qubit,
        in the order the X records were encountered.

    Raises:
        DeqExportError: If the X/Z record counts differ, or the records do
            not form a clean symplectic dual basis (each X anticommutes with
            exactly one Z and vice versa).
    """
    x_ops = [r for r in logical_ops if r["type"] == "X"]
    z_ops = [r for r in logical_ops if r["type"] == "Z"]
    if len(x_ops) != len(z_ops):
        raise DeqExportError(
            f"logical_ops has {len(x_ops)} X-type and {len(z_ops)} Z-type "
            "records; expected equal counts (one of each per logical qubit)."
        )
    if not x_ops:
        return []

    data_indices = sorted(data_indices)
    col = {q: i for i, q in enumerate(data_indices)}
    n = len(data_indices)

    x_mat = np.stack([_to_symplectic(r, col, n) for r in x_ops])
    z_mat = np.stack([_to_symplectic(r, col, n) for r in z_ops])
    anticommutes = check_commutativity(x_mat, z_mat)

    if not (anticommutes.sum(axis=0) == 1).all() or not (anticommutes.sum(axis=1) == 1).all():
        raise DeqExportError(
            "logical X/Z records do not form a clean symplectic dual basis "
            f"(expected each X to anticommute with exactly one Z and vice versa):\n{anticommutes}"
        )

    partner = anticommutes.argmax(axis=1)
    return [(x_ops[i], z_ops[int(partner[i])]) for i in range(len(x_ops))]
