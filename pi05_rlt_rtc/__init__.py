"""pi05-rlt-rtc: queue-aware RLT actor-critic + inference-time RTC integration package.

This package is a *thin adapter* around the pinned RLinf pi0.5/RLT stack. It
does not reimplement the VLA backbone, the RLT token transformer or the
distributed training framework. See ``docs/ARCHITECTURE.md`` for the mapping
between module paths and the implementation guide sections.
"""

__version__ = "0.1.0"

# The schema version stamped on every plan, transition, replay buffer and
# checkpoint produced by this package.  Bump it whenever the on-disk or
# in-flight layout changes in a way old consumers cannot interpret.
SCHEMA_VERSION = "pi05-rlt-rtc.schema.v1"

# Version of the *contract* between the feature service, the actor input
# layout and the delayed transition assembler.  A consumer must reject data
# produced with a different contract version.
PLANNING_STATE_VERSION = "queue-planning-state.v1"
