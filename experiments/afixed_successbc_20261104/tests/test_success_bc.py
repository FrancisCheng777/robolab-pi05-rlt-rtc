import dataclasses
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pytest

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.deployment.success_bc import success_execution_targets


@dataclass
class Row:
    capture_tick: int
    plan_id: int
    submitted_plan: np.ndarray
    executed_trace: np.ndarray
    executed_source_plan_ids: np.ndarray
    submitted_valid_mask: np.ndarray
    bc_target_plan: np.ndarray
    bc_target_valid_mask: np.ndarray
    planning_state: object
    teacher_source: str = 'teacher'
    episode_id: str = 'episode'
    def is_learnable(self): return True


def example():
    # D=C=2: the first window executes old pending commands, the second executes U0.
    state = object()
    first = Row(0, 10, np.full((2, 8), .3), np.full((2, 8), -.2), np.array([9, 9]),
                np.ones((2, 8), bool), np.zeros((2, 8)), np.ones((2, 8), bool), state)
    # Ends after executing only the first row of U1 at tick 4.
    last = Row(2, 11, np.full((2, 8), .6), np.array([[.3]*8, [.3]*8, [.6]*8]), np.array([10, 10, 11]),
               np.ones((2, 8), bool), np.zeros((2, 8)), np.ones((2, 8), bool), state)
    return [first, last], {'trial_id': 'episode', 'outcome': 'success', 'eligible': True}, SimpleNamespace(chunk=2, delay=2)


def test_delayed_actions_and_terminal_mask_without_mutating_sources():
    rows, label, contract = example()
    result, audit = success_execution_targets(rows, label, contract)
    np.testing.assert_allclose(result[0].bc_target_plan, .3)
    np.testing.assert_allclose(result[1].bc_target_plan[0], .6)
    assert result[0].bc_target_valid_mask.all()
    assert result[1].bc_target_valid_mask[0].all() and not result[1].bc_target_valid_mask[1].any()
    assert audit['executed_action_rows'] == 3 and audit['masked_action_rows'] == 1
    assert not rows[0].bc_target_plan.any()
    assert result[0].planning_state is rows[0].planning_state
    assert result[0].submitted_plan is rows[0].submitted_plan
    assert result[0].executed_trace is rows[0].executed_trace


def test_failure_bc_is_unchanged():
    rows, label, contract = example(); label['outcome'] = 'failure'
    result, audit = success_execution_targets(rows, label, contract)
    assert all(a is b for a, b in zip(rows, result))
    assert audit['changed_transitions'] == 0


@pytest.mark.parametrize('corruption', ['source', 'action', 'episode', 'nonfinite'])
def test_invalid_execution_alignment_rejected(corruption):
    rows, label, contract = example()
    if corruption == 'source': rows[1].executed_source_plan_ids[0] = 99
    if corruption == 'action': rows[1].executed_trace[0, 0] = .9
    if corruption == 'episode': rows[1].episode_id = 'another'
    if corruption == 'nonfinite': rows[1].executed_trace[0, 0] = np.nan
    with pytest.raises(ContractError): success_execution_targets(rows, label, contract)
