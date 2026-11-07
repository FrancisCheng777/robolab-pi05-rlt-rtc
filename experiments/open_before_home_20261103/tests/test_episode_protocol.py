import numpy as np
import pytest
from pi05_rlt_rtc.deployment.episode import EpisodeSignals
from pi05_rlt_rtc.contracts.errors import ContractError


def test_scoring_stops_episode_and_switch_is_idempotent():
    signals=EpisodeSignals(max_approach_ticks=100,max_critical_ticks=30)
    signals.poll(['b'],7)
    signals.poll(['b'],9)
    assert signals.critical_start_tick==7
    signals.poll(['c','a'],12)
    assert signals.outcome=='success' and signals.terminal_tick==11
    signals.poll(['a'],13)
    assert signals.outcome=='success'


def test_timeout_is_normal_failure_but_quit_is_aborted():
    signals=EpisodeSignals(max_approach_ticks=20,max_critical_ticks=5)
    signals.poll(['b'],2);signals.poll([],7)
    assert signals.outcome=='failure' and signals.reason=='critical_timeout'
    signals=EpisodeSignals();signals.poll(['q'],3)
    assert signals.outcome=='aborted'


def test_scores_before_execution_are_not_successes():
    signals=EpisodeSignals();signals.poll(['c'],0)
    assert signals.outcome=='aborted'
