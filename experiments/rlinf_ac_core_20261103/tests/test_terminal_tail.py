"""A human terminal event can arrive while the next request is still in flight."""
import copy
import pytest
from pi05_rlt_rtc.contracts.spaces import HorizonContract
from pi05_rlt_rtc.learning.transition_assembler import DelayedTransitionAssembler
from tests.test_learner import _collect,_TerminatingEnv
from tests.helpers import make_zero_state_encoder


def assemble(mutate=None):
    contract=HorizonContract(10,5,5)
    _,log,spec,_=_collect(contract,latency=4,num_ticks=28,env=_TerminatingEnv(terminate_at_step=28))
    valid={e['request_id'] for e in log.events if e['kind']=='plan' and e.get('accepted')}
    events=copy.deepcopy([e for e in log.events if e['kind']!='request' or e['request'].request_id in valid])
    for e in events:
        if e['kind']=='tick':e['reward']=1. if e['tick']==27 else 0.
    if mutate:events=mutate(events)
    assembler=DelayedTransitionAssembler(contract=contract,
        state_encoder=make_zero_state_encoder(spec,contract),gamma_per_tick=.99)
    assembler.ingest_run(events)
    return assembler.finalize()


def test_terminal_after_chunk_boundary_uses_only_actual_committed_tail():
    transitions=assemble();terminal=[t for t in transitions if t.terminated and t.is_learnable()]
    assert len(terminal)==1
    t=terminal[0]
    assert t.capture_tick==20 and t.duration_ticks==8
    assert t.next_planning_state is None and not t.bootstrap_allowed
    assert t.reward_sum==pytest.approx(.99**7)
    assert t.executed_source_plan_ids[5:].tolist()==[t.plan_id]*3
    assert t.primitive_rewards.tolist()==[0,0,0,0,0,0,0,1]


def test_missing_or_different_tail_commands_are_not_fabricated():
    def missing(events):return [e for e in events if not(e['kind']=='tick' and e['tick']==26)]
    def different(events):
        for e in events:
            if e['kind']=='tick' and e['tick']==26:e['source_plan_id']=999
        return events
    for mutate in (missing,different):
        assert not any(t.terminated and t.is_learnable() for t in assemble(mutate))
