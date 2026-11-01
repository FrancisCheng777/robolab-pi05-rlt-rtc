"""Official optimizer scale with the existing local update schedule; CPU only."""
from types import SimpleNamespace
import copy
import torch
from pi05_rlt_rtc.contracts.spaces import HorizonContract
from pi05_rlt_rtc.deployment.training import QueueLearner
from pi05_rlt_rtc.learning.replay import ReplayBuffer
from tests.test_learner import _collect


def learner_and_replay():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    _, _, spec, transitions = _collect(contract, num_ticks=30)
    replay = ReplayBuffer(100, feature_version_id='feature-v1',
                          norm_stats_sha256=transitions[0].norm_stats_sha256)
    for transition in transitions:
        replay.push(transition)
    cfg = SimpleNamespace(contract=contract, raw={'training': {
        'batch_size': 4,
        'actor_clip_grad': 10., 'critic_clip_grad': 10.,
        'lambda_q': .1, 'lambda_bc': 5., 'reference_dropout': .5}})
    return QueueLearner(cfg, spec, device='cpu', seed=11), replay


def test_actor_and_critic_keep_one_to_one_schedule_across_calls():
    learner, replay = learner_and_replay()
    learner.update(replay, 1)
    actor = copy.deepcopy(learner.actor.state_dict())
    critic = copy.deepcopy(learner.critic.online.state_dict())
    learner.update(replay, 3)
    assert any(not torch.equal(actor[k], v) for k, v in learner.actor.state_dict().items())
    assert any(not torch.equal(critic[k], v) for k, v in learner.critic.online.state_dict().items())
    assert learner.counts['updates'] == 4
    for optimizer in (learner.actor_opt, learner.critic_opt):
        assert {int(state['step']) for state in optimizer.state.values()} == {4}


def test_gradient_clipping_obeys_configured_scale():
    p = torch.nn.Parameter(torch.tensor(0.))
    optimizer = torch.optim.SGD([p], lr=1.)
    QueueLearner._step(100 * p, optimizer, [p], clip_grad=10.)
    assert torch.isclose(p, torch.tensor(-10.), atol=1e-5)


def test_production_settings_reach_learner_and_keep_discount():
    from pi05_rlt_rtc.config.runtime_config import load_runtime_config
    from pi05_rlt_rtc.learning.planning_state import PlanningStateSpec
    cfg = load_runtime_config('configs/runtime_franka.yaml')
    learner = QueueLearner(cfg, PlanningStateSpec.from_contract(cfg.contract, z_dim=16),
                           device='cpu', seed=1)
    assert cfg.raw['training']['gamma_per_tick'] == .99
    assert learner.actor_opt.param_groups[0]['lr'] == 3e-4
    assert learner.critic_opt.param_groups[0]['lr'] == 3e-4
    assert learner.settings['batch_size'] == 256
    assert learner.settings['lambda_bc'] == 5.
    assert learner.settings['reference_dropout'] == .5
