"""Human RLT critical-phase episode protocol; no device dependencies."""
from dataclasses import dataclass

@dataclass
class EpisodeSignals:
    max_approach_ticks: int = 900
    max_critical_ticks: int = 450
    critical_start_tick: int | None = None
    outcome: str | None = None
    terminal_tick: int | None = None
    reason: str | None = None

    def poll(self, keys, tick):
        if self.outcome is not None:
            return
        keys = list(keys)
        urgent = next((k for k in ('q', 'x', 'e') if k in keys), None)
        for key in ([urgent] if urgent else keys):
            if key in ('q','x','e'):
                self.outcome, self.reason = 'aborted', {'q':'operator_quit','x':'operator_abort','e':'operator_home'}[key]
            elif key == 'b' and self.critical_start_tick is None:
                self.critical_start_tick = tick
            elif key in ('a','c'):
                self.outcome = 'success' if key == 'c' else 'failure'
                self.reason = 'human_terminal'
            if self.outcome:
                break
        if not self.outcome:
            start = self.critical_start_tick
            if start is not None and tick-start >= self.max_critical_ticks:
                self.outcome,self.reason='failure','critical_timeout'
            elif start is None and tick >= self.max_approach_ticks:
                self.outcome,self.reason='failure','approach_timeout'
        if self.outcome:
            self.terminal_tick=tick-1 if tick else None
            if not tick and self.reason not in ('operator_quit', 'operator_abort', 'operator_home'):
                self.outcome,self.reason='aborted','no_executed_action'
