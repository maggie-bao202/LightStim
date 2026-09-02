"""Tests for Magic-H6 [[6,2,2]] and concatenated [[36,4,4]] builders."""

from __future__ import annotations

import pytest

from lightstim.protocols.magic_h6_benchmark import (
    build_h6_circuit,
    inject_noise,
    run_simulation,
    run_simulation_level2,
)


def test_build_is_noiseless_and_well_formed():
    circ, info, system = build_h6_circuit(level=1)
    assert info["code"] == "[[6,2,2]]"
    assert info["k"] == 2
    assert circ.num_observables == 2
    assert circ.num_detectors == 7           # 4 SE + 1 Bell-pair H-check + 2 final X-stab
    dets, obs = circ.compile_detector_sampler().sample(1024, separate_observables=True)
    assert not dets.any()
    assert not obs.any()
    circ.detector_error_model(decompose_errors=True)   # graphlike


def test_build_level_2_is_framework_native_and_noiseless():
    circ, info, system = build_h6_circuit(level=2)
    assert info["code"] == "[[36,4,4]]"
    assert info["k"] == 4
    assert circ.num_qubits == system.num_qubits
    # Built through CircuitBuilder + SyndromeTracker: auto-generated detectors
    # for every [[6,2,2]] stabilizer on every block, plus block-logical observables.
    assert circ.num_detectors > 0
    assert circ.num_observables >= 4
    assert len(info["target_observable_indices"]) >= 1
    assert set(info["target_observable_indices"]).isdisjoint(info["ps_observable_indices"])
    dets, obs = circ.compile_detector_sampler().sample(256, separate_observables=True)
    assert not dets.any()
    assert not obs.any()


def test_build_level_2_se_between_layers_still_noiseless():
    circ, info, _ = build_h6_circuit(level=2, se_between_layers=True)
    # Extra SE rounds -> strictly more detectors, still clean.
    assert info["num_detectors"] > build_h6_circuit(level=2)[1]["num_detectors"]
    dets, obs = circ.compile_detector_sampler().sample(128, separate_observables=True)
    assert not dets.any()
    assert not obs.any()


def test_invalid_level_raises_value_error():
    with pytest.raises(ValueError):
        build_h6_circuit(level=3)


def test_run_simulation_level2_noiseless_is_perfect():
    circ, info, system = build_h6_circuit(level=2)
    stats = run_simulation_level2(
        circ,
        p=0.0,
        info=info,
        mode="full",
        data_indices=list(system.data_indices),
        num_samples=512,
    )
    assert stats["shots"] == 512
    assert stats["accepted"] == 512
    assert stats["failed"] == 0
    assert stats["post_selection_rate"] == 1.0
    assert stats["logical_error_rate"] == 0.0


def test_run_simulation_level2_smoke_with_noise():
    circ, info, system = build_h6_circuit(level=2)
    stats = run_simulation_level2(
        circ,
        p=1e-3,
        info=info,
        mode="full",
        data_indices=list(system.data_indices),
        num_samples=4096,
    )
    assert stats["shots"] == 4096
    assert 0 <= stats["accepted"] <= 4096
    assert 0 <= stats["failed"] <= stats["accepted"]


def test_run_simulation_level2_scores_far_below_level1():
    # Regression guard for the observable-split fix: run_simulation_level2 must
    # count only the 4 genuine [[36,4,4]] logicals, not first-order stabilizer
    # excitations. Pre-fix this LER was ~1.7e-2 at p=2e-3 (>= the level-1 rate);
    # with the correct split it is ~1e-4 (quadratic distillation gain).
    circ, info, system = build_h6_circuit(level=2)
    assert len(info["target_observable_indices"]) == 4
    stats = run_simulation_level2(
        circ,
        p=2e-3,
        info=info,
        mode="full",
        data_indices=list(system.data_indices),
        num_samples=200_000,
        batch_size=50_000,
    )
    assert stats["accepted"] > 1_000
    assert stats["logical_error_rate"] < 5e-3


def test_run_simulation_level2_max_errors_early_stop():
    # The level-1-style early-stop: with max_errors set, sampling stops on the
    # first batch boundary after that many surviving shots have failed. At
    # p=3e-3 acceptance is ~2.5% and the (correct) LER ~5e-4, so a 1e6-shot
    # budget accumulates failures many times over -> the cap always trips first.
    circ, info, system = build_h6_circuit(level=2)
    stats = run_simulation_level2(
        circ,
        p=3e-3,
        info=info,
        mode="full",
        data_indices=list(system.data_indices),
        num_samples=1_000_000,
        batch_size=50_000,
        max_errors=1,
    )
    assert stats["failed"] >= 1               # stopped because the cap was hit
    assert stats["shots"] < 1_000_000         # ... not because the budget ran out
    assert stats["shots"] % 50_000 == 0       # stopped on a batch boundary

    full = run_simulation_level2(
        circ, p=0.0, info=info, data_indices=list(system.data_indices),
        num_samples=4096, batch_size=1024, max_errors=5,
    )
    assert full["shots"] == 4096              # no failures -> samples the full cap
    with pytest.raises(ValueError):
        run_simulation_level2(circ, p=1e-3, info=info, num_samples=64, max_errors=0)


def test_inject_noise_modes():
    circ, _, system = build_h6_circuit(level=1)
    di = list(system.data_indices)
    full = inject_noise(circ, 1e-2, "full", data_indices=di)
    idle = inject_noise(circ, 1e-2, "idle", data_indices=di)
    assert full.num_detectors == circ.num_detectors
    assert idle.num_detectors == circ.num_detectors
    # idle mode adds DEPOLARIZE1 at SE_start -> strictly more instructions
    assert len(idle.flattened()) > len(full.flattened())
    with pytest.raises(ValueError):
        inject_noise(circ, 1e-2, "bogus")


@pytest.mark.slow
def test_run_simulation_post_selects_and_scores():
    circ, _, system = build_h6_circuit(level=1)
    di = list(system.data_indices)
    s = run_simulation(circ, 1e-2, mode="full", data_indices=di,
                       max_errors=50, max_shots=500_000,
                       batch_size=10_000, num_workers=4)
    assert s.shots > 0
    assert 0.0 < s.post_selection_rate < 1.0     # some shots rejected
    assert 0.0 < s.logical_error_rate < 1.0


@pytest.mark.slow
def test_run_simulation_level2_post_selects_and_scores():
    circ, info, system = build_h6_circuit(level=2)
    di = list(system.data_indices)
    s = run_simulation_level2(circ, 1e-3, info=info, mode="full",
                              data_indices=di, num_samples=200_000)
    assert 0.0 < s["post_selection_rate"] < 1.0   # detectors reject some shots
    assert 0.0 <= s["logical_error_rate"] < 1.0
