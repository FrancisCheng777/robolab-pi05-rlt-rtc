# Provenance and licensing notes

This repository is an **integration implementation** written against the
research snapshot in `docs/spec/`. It does not claim authorship of π0.5, RLT or
inference-time RTC, and it ships no model weights or datasets.

- π0.5 / openpi — Physical Intelligence, Apache-2.0:
  https://github.com/Physical-Intelligence/openpi (pinned `215abfb2…`).
- RLT (RL Token) — research code referenced through RLinf; see the paper links
  in `docs/spec/PI05_RLT_RTC_IMPLEMENTATION_GUIDE.md`.
- Inference-time RTC / real-time-chunking-kinetix — Physical Intelligence:
  https://github.com/Physical-Intelligence/real-time-chunking-kinetix
  (pinned `9296f31d…`). The `full_vjp` port in `pi05_rlt_rtc/rtc/` follows that
  reference math and credits it in the module docstrings.
- RLinf — main upstream baseline:
  https://github.com/RLinf/RLinf (pinned `c70606f0…`).

`pi05_rlt_rtc/rtc/approx.py` is a behavioural port of the pinned RLinf
approximate RTC sampler; it retains the upstream semantics and is referenced as
"RLinf approximate RTC" wherever it is reported.

New code in this repository is provided under Apache-2.0. See the upstream
repositories for their own license terms; asset licenses (weights, data) must
be checked separately.

## Physical hardware integration (2026-10-29)

Panda/Robotiq bridge and Kinect/D405 camera code were adapted from the user-owned `heinrich:/home/heinrich/pi05_plug_demo/` snapshot. The Robotiq driver retains its attribution to ros-industrial/robotiq. Stage-1 YAML derives from the pinned RLinf SFT example under Apache-2.0; the physical profile and explicit loader registration are local additions.

## Current RoboLab integration (2026-11-01)

The accepted Panda/Robotiq control and D405 acquisition are reused read-only
from FrancisCheng777/robolab-collecting-pipeline at
`eb9a6c5855a26025841204c715880f4ea3182709`. Its Robotiq SDK attribution and
licenses remain in the submodule. The bridge calibration is not modified.
The legacy Kinect/dual-camera snapshots above are not the active data path.
The queue-aware learning, episode coordinator, v2 interface and CPU owner are
local adaptations. Reward defaults follow the original PI RLT paper (terminal
0/1), rather than the generic RLinf signed keyboard wrapper.
