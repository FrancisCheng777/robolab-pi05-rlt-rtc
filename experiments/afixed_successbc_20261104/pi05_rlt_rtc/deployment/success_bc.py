"""Success-only BC targets aligned to executed source-plan IDs and RTC ticks."""
import dataclasses
import numpy as np

from pi05_rlt_rtc.contracts.errors import ContractError


def success_execution_targets(transitions, annotation, contract):
    rows = list(transitions)
    audit = {'trial_id': annotation['trial_id'], 'outcome': annotation['outcome'],
             'changed_transitions': 0, 'executed_action_rows': 0, 'masked_action_rows': 0}
    if annotation['outcome'] != 'success':
        return rows, audit
    if not annotation.get('eligible'):
        raise ContractError('success BC requires an eligible labelled episode')
    executed = {}
    for tr in rows:
        if tr.episode_id != annotation['trial_id']:
            raise ContractError('success BC cannot mix episodes')
        if not tr.is_learnable():
            continue
        for offset, (action, source) in enumerate(zip(tr.executed_trace, tr.executed_source_plan_ids)):
            tick = tr.capture_tick + offset
            action = np.asarray(action, dtype=np.float32)
            if action.shape != (8,) or not np.isfinite(action).all():
                raise ContractError('invalid executed BC action')
            if tick in executed:
                old_source, old_action = executed[tick]
                if old_source != int(source) or not np.array_equal(old_action, action):
                    raise ContractError('conflicting execution records at the same tick')
            executed[tick] = (int(source), action.copy())
    result = []
    for tr in rows:
        if not tr.is_learnable():
            result.append(tr)
            continue
        target = np.zeros_like(tr.submitted_plan, dtype=np.float32)
        mask = np.zeros_like(tr.submitted_valid_mask, dtype=bool)
        for index in range(contract.chunk):
            tick = tr.capture_tick + contract.delay + index
            record = executed.get(tick)
            if record is None:
                continue  # Terminal tails which never executed must not receive BC.
            source, action = record
            valid = np.asarray(tr.submitted_valid_mask[index], dtype=bool)
            if valid.any() and source != tr.plan_id:
                raise ContractError('RTC BC target source does not match its submitted plan')
            if not np.allclose(action[valid], tr.submitted_plan[index][valid], atol=1e-6, rtol=1e-6):
                raise ContractError('RTC BC target differs from the executed submitted action')
            target[index] = action
            mask[index] = valid
        audit['changed_transitions'] += 1
        audit['executed_action_rows'] += int(mask.any(axis=1).sum())
        audit['masked_action_rows'] += int((~mask.any(axis=1)).sum())
        result.append(dataclasses.replace(tr, bc_target_plan=target, bc_target_valid_mask=mask,
                                          teacher_source='successful_executed_policy_rtc_v1'))
    return result, audit
