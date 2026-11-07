import json
from types import SimpleNamespace
import pytest
from pi05_rlt_rtc.deployment import stage2_session as s

@pytest.mark.parametrize('mode,errors,allowed', [('Idle',[],True),('Reflex',[],False),('Idle',['communication_constraints_violation'],False),('User stopped',[],False),('Move',[],False)])
def test_read_only_preflight_requires_idle_without_errors(monkeypatch,mode,errors,allowed):
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0,stderr='',stdout=json.dumps({'robot_mode':mode,'current_errors':errors,'last_motion_errors':['communication_constraints_violation']}))
    monkeypatch.setattr(s.subprocess,'run',run)
    if allowed:assert s.require_robot_idle()['robot_mode']=='Idle'
    else:
        with pytest.raises(RuntimeError,match='Franka Desk'):s.require_robot_idle()
    assert calls==[['echo_robot_state','172.16.0.40']]

def test_no_valid_robot_state_is_not_ready(monkeypatch):
    monkeypatch.setattr(s.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stderr='',stdout='not a state'))
    with pytest.raises(RuntimeError,match='有效数据'):s.require_robot_idle()
