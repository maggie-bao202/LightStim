"""Quantinuum Magic-H6 ``|H>`` magic-state distillation, in the LightStim framework.

Ports both the level-1 single-block ``[[6, 2, 2]]`` (k = 2) protocol and
the level-2 concatenated ``[[36, 4, 4]]`` (k = 4) construction from
``Logical-Magic-State-Distillation-Circuits`` (itself a port of
https://github.com/CQCL/Magic-H6). The base code lives in
``lightstim.qec_code.six_two_two``; this module drives one block through a
distillation round and scores it with the SyndromeTracker's auto-generated
detectors and observables instead of the raw port's hand-indexed
``checks()`` / ``success()``.

What the level-1 round does (mirrors ``get_dist_circ`` in CQCL/Magic-H6)
----------------------------------------------------------------------
1. Prepare the ``[[6, 2, 2]]`` codeword ``|++>_L`` with the verbatim 10-gate
    Magic-H6 encoder (``get_dist_circ`` -- ``get_dist_circ``'s encoder
   block).
2. One native syndrome-extraction round: every ``[[6, 2, 2]]``
   stabilizer becomes a DETECTOR, so a single fault that disturbs a stabilizer
   is heralded. (A LightStim strengthening; the raw port has no native rounds.)
3. The Bell-pair "H-check" (``SixTwoTwoLogicalXCheckBlock`` --
   ``get_dist_circ``'s ancilla block): a non-destructive readout of ``X0_L`` and
   ``X1_L`` onto a Bell pair. Both are deterministic on ``|++>_L``, so the
   tracker emits DETECTORs that herald encoder faults which flip a logical
   without touching a stabilizer (invisible to step 2).
4. Destructive X-basis readout. The tracker emits the two logical observables
   ``X0_L = X0 X2 X4`` and ``X1_L = X1 X3 X5``; a distilled block has both = 0.

Distillation = **post-select on every detector** (any syndrome or H-check flip
-> discard, the ``checks()`` step) and measure the residual logical-flip rate on
the two observables (the ``success()`` step).

Fault tolerance
---------------
The encoder in step 1 is *not* fault-tolerant (same as the raw port's
non-FT benchmark path); the H-check in step 3 heralds its logical-flipping
faults. The flag-verified FT encoder for ``|00>_L`` (Fig. 5 of arXiv:2506.14688,
``Code614.py``) is ported as ``lightstim.qec_code.six_two_two.get_ft_init_circ``;
the level-2 builder here uses only its data-CX core (:func:`_zero_zero_l_encoder`)
-- see the level-2 "Known limitations" below.

Public API
----------
build_h6_circuit(level=1)                       -> (circuit, info, system)
inject_noise(circuit, p, mode, data_indices)   -> noisy circuit
run_simulation(circuit, p, ...)                -> SimulationStats           (level 1)
run_simulation_level2(circuit, p, info, ...)   -> dict with accept/fail metrics (level 2)

Level 2 (concatenated ``[[36, 4, 4]]``, k = 4)
---------------------------------------------
``build_h6_circuit(level=2)`` builds the concatenated construction the same way
``tg_distillation`` builds its 7-to-1 factory: 20 ``[[6, 2, 2]]`` patches in one
``QECSystem`` (6 data blocks ``b0..b5``, 2 aux ``aux0/aux1``, 12 resource
``res0..res11``), every gate applied through ``SixTwoTwoLogicalOpSet`` +
``CircuitBuilder``, one native SE round after the encoders. Every ``[[6, 2, 2]]``
stabilizer on every block becomes a tracker-generated DETECTOR (post-selection),
and the block logicals are threaded into ``OBSERVABLE_INCLUDE`` records. Every
resource / aux qubit is measured once, at the terminal ``apply_data_readout``;
the byproduct frame update is folded into the observable definitions by the
tracker. No hand-indexed ``checks()`` / ``success()`` bit-slicing and no
mid-circuit classically-controlled correction.

``run_simulation_level2`` scores it with the ``tg_distillation`` post-selection
pattern: discard any shot with a firing detector, GF(2)-split the tracker
observables into an output set and a post-select set with
``lightstim.simulation.observable_analysis.identify_distillation_observables``,
discard shots whose post-select observables are nonzero, and count a surviving
shot as failed if any output observable is 1.

Known limitations (follow-ups, tracked in the PR description):

* The tracker exposes the six data blocks' logical DOF as a mixed 8-dimensional
  observable subspace, not the clean 4 level-2 logicals + 8 level-2 stabilizers
  split. ``identify_distillation_observables`` therefore separates *outer-code*
  support, not the exact level-2 logical operators, so ``logical_error_rate`` is
  a conservative upper bound on the distilled ``|H>`` infidelity rather than the
  paper's figure of merit. A faithful version needs the explicit level-2
  logical / stabilizer observables built from ``SixTwoTwoCode._X_CHECKS`` /
  ``_X_LOGICALS``.
* Input ``|H>`` infidelity is modelled only by circuit-level depolarizing ``p``;
  there is no separate ``p_in`` injection / calibration (cf.
  ``tg_distillation.estimate_p_in``), so the run does not yet demonstrate the
  quadratic ``p_in -> p_in^2`` suppression.
* The FT flag-verified ``|00>_L`` encoder (Fig. 5 of arXiv:2506.14688) is
  reduced to its data-CX core (:func:`_zero_zero_l_encoder`); flag verification
  needs a non-destructive mid-circuit ancilla-measurement gadget the builder
  does not expose.
"""

from __future__ import annotations

import numpy as np
import stim

from lightstim.qec_code.six_two_two import (
    SixTwoTwoCode,
    SixTwoTwoExtractionBlock,
    SixTwoTwoLogicalOpSet,
    SixTwoTwoLogicalXCheckBlock,
    get_dist_circ,
)
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.tracker import SyndromeTracker
from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector
from lightstim.simulation.decoder_backend.pipeline import SimulationPipeline
from lightstim.simulation.decoder_backend.config import DecoderConfig
from lightstim.simulation.observable_analysis import (
    build_obs_patch_matrix,
    identify_distillation_observables,
    transform_observables,
)

PATCH_NAME = "c622"

# Level-2 patch roster and grid layout.
_L2_DATA_BLOCKS = tuple(f"b{i}" for i in range(6))
_L2_AUX_BLOCKS = ("aux0", "aux1")
_L2_RESOURCE_BLOCKS = tuple(f"res{i}" for i in range(12))
_L2_PATCH_NAMES = _L2_DATA_BLOCKS + _L2_AUX_BLOCKS + _L2_RESOURCE_BLOCKS
_L2_FT_INIT_BLOCKS = ("b2", "b3", "b4", "b5", "aux0", "aux1")
_L2_LAYOUT_PITCH = 20


def _data_by_label(system: QECSystem, name: str) -> list[int]:
    """Global data-qubit indices of patch ``name``, in code-label order 0..5."""
    return sorted(
        q for q in system.data_indices if system.index_to_owner_map.get(q) == name
    )


def _zero_zero_l_encoder(data: list[int]) -> stim.Circuit:
    """``|00>_L`` encoder for the ``[[6,2,2]]`` code (data-CX core of ``Code614.py``).

    This is ``get_ft_init_circ`` with the two flag ancillas and their four
    verification CX gates removed -- the logical state it prepares is identical
    (the flag CX pair cancels on the data). Flag-verified FT preparation (Fig. 5
    of arXiv:2506.14688) needs a flag ancilla measured mid-circuit between the
    entangling and disentangling CX layers, which the current ``CircuitBuilder``
    does not expose as a non-destructive gadget; it is a documented follow-up
    (see module docstring). Same fault-tolerance status as the level-1 encoder.
    """
    s = list(data)
    c = stim.Circuit()
    c.append("H", [s[0], s[2]])
    c.append("CX", [s[0], s[1]])
    c.append("CX", [s[2], s[3]])
    c.append("CX", [s[0], s[4]])
    c.append("CX", [s[2], s[5]])
    c.append("CX", [s[0], s[5]])
    c.append("CX", [s[2], s[4]])
    return c


def _build_h6_level2_circuit(
    se_between_layers: bool = False,
) -> tuple[stim.Circuit, dict, QECSystem]:
    """Build the level-2 concatenated ``[[36,4,4]]`` Magic-H6 circuit (noiseless).

    Structured like ``lightstim.protocols.tg_distillation.build_distillation_circuit``:
    one ``QECSystem`` of 20 ``[[6, 2, 2]]`` patches, every gate through
    ``SixTwoTwoLogicalOpSet`` + ``CircuitBuilder``.

    Args:
        se_between_layers: if True, run an extra native SE round after every
            Clifford layer (stronger heralding, but every added round multiplies
            the post-selection cost -- acceptance collapses well before
            ``p = 5e-3``). Default False: a single SE round after the encoders,
            matching the reference protocol's per-block stabilizer checks.

    Returns ``(circuit, info, system)`` where ``info`` carries the GF(2) observable
    transform (``obs_transform``) and the target / post-select observable index
    lists used by :func:`run_simulation_level2`.
    """
    system = QECSystem()
    gp = {
        name: system.add_patch(
            SixTwoTwoCode(),
            offset=(
                _L2_LAYOUT_PITCH * (slot % 5),
                _L2_LAYOUT_PITCH * (slot // 5),
            ),
            name=name,
        )
        for slot, name in enumerate(_L2_PATCH_NAMES)
    }

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    blocks = [_data_by_label(system, name) for name in _L2_DATA_BLOCKS]
    aux = [_data_by_label(system, name) for name in _L2_AUX_BLOCKS]
    resources = [_data_by_label(system, name) for name in _L2_RESOURCE_BLOCKS]

    ops = SixTwoTwoLogicalOpSet()

    def se_round() -> None:
        block = SixTwoTwoExtractionBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=block.circuit, rounds=1)

    def layer_se_round() -> None:
        if se_between_layers:
            se_round()

    # 1. |0> on every data qubit.
    all_data = [q for group in (blocks + aux + resources) for q in group]
    builder.initialize({q: "Z" for q in all_data}, n=system.num_qubits)

    # 2. Encoders (unitary blocks): b0,b1 + all resources via get_dist_circ -> |++>_L;
    #    b2..b5,aux via the |00>_L encoder.
    builder.apply_unitary_block(get_dist_circ(blocks[0]))
    builder.apply_unitary_block(get_dist_circ(blocks[1]))
    for name in _L2_FT_INIT_BLOCKS:
        builder.apply_unitary_block(_zero_zero_l_encoder(_data_by_label(system, name)))
    for res in resources:
        builder.apply_unitary_block(get_dist_circ(res))

    # 3. Init SE round: every [[6,2,2]] stabilizer on every block -> DETECTOR.
    se_round()

    # 4. Clifford network before the first Ry stage (Code614.py order).
    ops.transversal_hadamard(builder, gp["b2"])
    ops.transversal_cnot(builder, gp["b2"], gp["b3"])
    ops.transversal_hadamard(builder, gp["b4"])
    ops.transversal_cnot(builder, gp["b4"], gp["b5"])
    ops.transversal_cnot(builder, gp["b2"], gp["b0"])
    ops.transversal_cnot(builder, gp["b3"], gp["b1"])
    ops.transversal_cnot(builder, gp["b0"], gp["b4"])
    ops.transversal_cnot(builder, gp["b1"], gp["b5"])
    ops.transversal_cnot(builder, gp["b4"], gp["b2"])
    ops.transversal_cnot(builder, gp["b5"], gp["b3"])
    layer_se_round()

    # 5. Ry(-pi/2)_L x6: CY(res_i -> b_i), then rotate the resource (S_DAG, H).
    #    The resource is consumed at the terminal readout (deferred, exact).
    for i in range(6):
        ops.transversal_cy(builder, gp[f"res{i}"], gp[f"b{i}"])
    rot1 = stim.Circuit()
    for i in range(6):
        rot1.append("S_DAG", resources[i])
        rot1.append("H", resources[i])
    builder.apply_unitary_block(rot1)
    layer_se_round()

    # 6. Logical-Z extraction into the aux blocks (Code614.py order).
    ops.transversal_hadamard(builder, gp["aux0"])
    ops.transversal_cnot(builder, gp["aux0"], gp["aux1"])
    ops.transversal_cz(builder, gp["aux0"], gp["b0"])
    ops.transversal_cz(builder, gp["aux1"], gp["b1"])
    ops.transversal_cz(builder, gp["aux0"], gp["b2"])
    ops.transversal_cz(builder, gp["aux1"], gp["b3"])
    ops.transversal_cz(builder, gp["aux0"], gp["b4"])
    ops.transversal_cz(builder, gp["aux1"], gp["b5"])
    ops.transversal_cnot(builder, gp["aux0"], gp["aux1"])
    ops.transversal_hadamard(builder, gp["aux0"])
    layer_se_round()

    # 7. Ry(+pi/2)_L x6 with the second six resources.
    for i in range(6):
        ops.transversal_cy(builder, gp[f"res{i + 6}"], gp[f"b{i}"])
    rot2 = stim.Circuit()
    for i in range(6):
        rot2.append("S_DAG", resources[i + 6])
        rot2.append("H", resources[i + 6])
    builder.apply_unitary_block(rot2)
    layer_se_round()

    # 8. One terminal destructive readout of everything.
    #    Data blocks: X (folds the reference's final transversal H + Z-readout).
    #    Aux / resource qubits: Z.
    final: dict[int, str] = {}
    for group in blocks:
        final.update({q: "X" for q in group})
    for group in aux + resources:
        final.update({q: "Z" for q in group})
    builder.apply_data_readout(final_measurements=final)

    circuit = builder.circuit

    # 9. Separate the level-2 output observables from the outer-code / byproduct
    #    observables via GF(2) elimination on the data-block columns.
    obs_matrix, patch_names = build_obs_patch_matrix(circuit, system)
    obs_transform, target_indices, ps_indices = identify_distillation_observables(
        obs_matrix, patch_names, list(_L2_DATA_BLOCKS)
    )

    info = {
        "level": 2,
        "code": "[[36,4,4]]",
        "k": 4,
        "rounds": 1,
        "num_qubits": circuit.num_qubits,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "obs_transform": obs_transform,
        "target_observable_indices": target_indices,
        "ps_observable_indices": ps_indices,
    }
    return circuit, info, system


def build_h6_circuit(level: int = 1, se_between_layers: bool = False):
    """Build the noiseless Magic-H6 distillation circuit.

    Level 1 is a single ``[[6, 2, 2]]`` block with one SE round; level 2 is the
    20-patch concatenated ``[[36, 4, 4]]`` construction (see
    :func:`_build_h6_level2_circuit`).

    Args:
        level:  Concatenation depth. ``1`` is a single ``[[6, 2, 2]]`` block
            (k = 2). ``2`` is concatenated ``[[36, 4, 4]]`` (k = 4).
        se_between_layers:  level 2 only -- run an extra SE round after every
            Clifford layer (see :func:`_build_h6_level2_circuit`). Default False.

    Returns:
        ``(circuit, info, system)`` where ``circuit`` is a clean ``stim.Circuit``.
        For level 2, ``info`` also carries ``obs_transform`` /
        ``target_observable_indices`` / ``ps_observable_indices`` for
        :func:`run_simulation_level2`.
    """
    if level not in (1, 2):
        raise ValueError(f"Unsupported level {level}. Choose 1 or 2.")

    if level == 2:
        return _build_h6_level2_circuit(se_between_layers=se_between_layers)

    rounds = 1  # [[6,2,2]] native syndrome-extraction rounds: fixed at 1.

    system = QECSystem()
    system.add_patch(SixTwoTwoCode(h_check_ancillas=2), name=PATCH_NAME)

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    data = _data_by_label(system, PATCH_NAME)

    # 1. |0>^6, then the verbatim Magic-H6 encoder -> |++>_L codeword.
    builder.initialize({q: "Z" for q in data}, n=system.num_qubits)
    builder.apply_unitary_block(get_dist_circ(data))

    # 2. Stabilizer check: native SE rounds ([[6,2,2]] heralds a disturbed stabilizer).
    se = SixTwoTwoExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=rounds)

    # 3. Bell-pair H-check: heralds encoder faults that flip a logical only.
    h_check = SixTwoTwoLogicalXCheckBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=h_check.circuit, rounds=1)

    # 4. Destructive X readout -> observables X0_L, X1_L + final stabilizer detectors.
    builder.apply_data_readout(final_measurements={q: "X" for q in data})

    circuit = builder.circuit
    info = {
        "level": 1,
        "code": "[[6,2,2]]",
        "k": 2,
        "rounds": rounds,
        "num_qubits": circuit.num_qubits,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
    }
    return circuit, info, system


def inject_noise(
    circuit: stim.Circuit,
    p: float,
    mode: str = "full",
    data_indices=None,
) -> stim.Circuit:
    """Inject noise into a clean Magic-H6 circuit.

    Args:
        p:            Circuit-level depolarizing / flip rate (all channels equal).
        mode:         ``"full"`` circuit-level noise on every operation, or
                      ``"idle"`` which additionally depolarizes idling data
                      qubits at ``p/5`` (the raw port's ``m = p/5`` memory term).
        data_indices: ``system.data_indices``; targets of the ``p_idle`` rule.
                      Defaults to every qubit (fine for ``mode="full"``).
    """
    idle_targets = (
        list(range(circuit.num_qubits)) if data_indices is None else list(data_indices)
    )
    if mode == "full":
        cfg = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p)
    elif mode == "idle":
        cfg = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p / 5)
    else:
        raise ValueError(f"Unknown noise mode {mode!r}. Choose 'full' or 'idle'.")
    return NoiseInjector.from_circuit_level(cfg, idle_targets).inject_noise(circuit)


def run_simulation(
    circuit: stim.Circuit,
    p: float,
    mode: str = "full",
    data_indices=None,
    max_shots: int = 20_000_000,
    max_errors: int = 200,
    batch_size: int = 20_000,
    num_workers: int = 8,
    print_progress: bool = False,
):
    """Run noisy Magic-H6 scoring for detector/observable-based circuits.

    Post-selects on **every** detector (any syndrome flip discards the shot) and
    reports the residual logical-flip rate on surviving shots.

    This helper is for the level-1 builder output (plain post-select on every
    detector). Level 2 needs the GF(2) observable transform + observable
    post-selection -- use ``run_simulation_level2`` with the ``info`` dict from
    ``build_h6_circuit(level=2)``.

    Returns a ``SimulationStats``; ``post_selection_rate`` is the acceptance
    rate and ``logical_error_rate`` is the distilled-state failure rate.
    """
    if circuit.num_detectors == 0 or circuit.num_observables == 0:
        raise ValueError(
            "run_simulation requires a detector/observable-annotated circuit."
        )

    noisy = inject_noise(circuit, p, mode, data_indices=data_indices)

    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("pymatching"),
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        num_workers=num_workers,
        post_select_detector_indices=list(range(noisy.num_detectors)),
        print_progress=print_progress,
    )
    return pipeline.run(noisy)


def run_simulation_level2(
    circuit: stim.Circuit,
    p: float,
    info: dict,
    mode: str = "full",
    data_indices=None,
    num_samples: int = 200_000,
    batch_size: int = 50_000,
) -> dict:
    """Score the level-2 ``[[36,4,4]]`` circuit by tracker detectors + observables.

    Pure post-selection (no decoding), the same shape as
    ``tg_distillation.run_simulation``'s non-identity branch:

    * discard any shot with a firing DETECTOR (every ``[[6,2,2]]`` stabilizer on
      every block),
    * GF(2)-transform the tracker observables (``info["obs_transform"]``) and
      discard any shot whose post-select observables
      (``info["ps_observable_indices"]``) are nonzero,
    * a surviving shot fails if any output observable
      (``info["target_observable_indices"]``) is 1.

    ``logical_error_rate`` is a **conservative upper bound** on the distilled
    ``|H>`` infidelity, not the paper's figure of merit -- see the module
    docstring's "Known limitations".

    Args:
        info: the dict returned alongside the circuit by ``build_h6_circuit(level=2)``.

    Returns the same keys as before: ``shots``, ``accepted``, ``failed``, ``p``,
    ``post_selection_rate``, ``logical_error_rate``.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1.")
    if circuit.num_detectors == 0 or circuit.num_observables == 0:
        raise ValueError(
            "run_simulation_level2 expects the detector/observable-annotated "
            "circuit from build_h6_circuit(level=2)."
        )

    obs_transform = np.asarray(info["obs_transform"], dtype=int)
    target_idx = list(info["target_observable_indices"])
    ps_idx = list(info["ps_observable_indices"])
    if not target_idx:
        raise ValueError(
            "No level-2 output observable was identified; the build is malformed."
        )

    noisy = inject_noise(circuit, p, mode, data_indices=data_indices)
    sampler = noisy.compile_detector_sampler()

    shots = 0
    accepted = 0
    failed = 0
    while shots < num_samples:
        take = min(batch_size, num_samples - shots)
        dets, obs = sampler.sample(take, separate_observables=True)
        shots += take

        obs_t = transform_observables(obs, obs_transform)
        keep = ~dets.any(axis=1)
        if ps_idx:
            keep &= ~obs_t[:, ps_idx].any(axis=1)
        if not keep.any():
            continue

        kept = obs_t[keep]
        accepted += int(keep.sum())
        failed += int(kept[:, target_idx].any(axis=1).sum())

    return {
        "shots": num_samples,
        "accepted": accepted,
        "failed": failed,
        "p": p,
        "post_selection_rate": accepted / num_samples,
        "logical_error_rate": (failed / accepted) if accepted else float("nan"),
    }
