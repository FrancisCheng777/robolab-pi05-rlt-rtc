# Executed test results

Generated (UTC): 2026-10-29T05:05:03.678422+00:00
Command: `PI05_TEST_LEDGER=... python -m pytest tests -v`
Log: `runs/physical_sync_4090/local_pytest.log`

| ID | status | environment | notes |
|---|---|---|---|
| T01 | SKIPPED | not_executed | Skipped: T01 requires the real LeRobot dataset; no dataset_audit can substitute for the local schema check (guide 5.2) / |
| T02 | SKIPPED | not_executed | Skipped: T02 requires the real two-camera dataset and the RLinf image transforms / real-environment prerequisite: real t |
| T03 | PASSED | cpu_unit_no_upstream |  |
| T04 | PASSED | cpu_unit_no_upstream |  |
| T05 | PASSED | cpu_unit_no_upstream |  |
| T06 | PASSED | cpu_unit_no_upstream |  |
| T07 | SKIPPED | not_executed | Skipped: T07 requires the real dataset plus the RLinf PandaOutputs transform to prove no double delta application / real |
| T08 | PASSED | cpu_unit_no_upstream |  |
| T09 | PASSED | cpu_unit_no_upstream |  |
| T10 | SKIPPED | not_executed | Skipped: T10 requires the real RLinf training/inference state preprocessing paths to be compared on the same raw state / |
| T11 | SKIPPED | not_executed | Skipped: T11 requires the real RLT encoder/decoder and VLA parameters / real-environment prerequisite: real RLT encoder/ |
| T12 | SKIPPED | not_executed | Skipped: T12 requires the real pi0.5 flow-matching loss / real-environment prerequisite: real pi0.5 flow-matching loss |
| T13 | SKIPPED | not_executed | Skipped: T13 requires the real causal RLT reconstruction decoder / real-environment prerequisite: real causal RLT recons |
| T14 | PASSED | cpu_unit_no_upstream |  |
| T15 | SKIPPED | not_executed | Skipped: T15 requires the real prefix padding masks from the tokenizer / real-environment prerequisite: real tokenizer p |
| T16 | SKIPPED | not_executed | Skipped: T16 requires a real Stage 1 run on real data / real-environment prerequisite: real Stage 1 run on real data |
| T17 | SKIPPED | not_executed | Skipped: T17 requires the real frozen pi0.5/RLT Stage 2 parameter checksums; the toy gradient-boundary audit is a differ |
| T18 | SKIPPED | not_executed | Skipped: T18 requires a real RLinf Stage 2 checkpoint loader / real-environment prerequisite: real RLinf Stage 2 checkpo |
| T19 | PASSED | cpu_unit_no_upstream |  |
| T20 | PASSED | cpu_unit_no_upstream |  |
| T21 | PASSED | cpu_unit_no_upstream |  |
| T22 | PASSED | cpu_unit_no_upstream |  |
| T23 | PASSED | cpu_unit_no_upstream |  |
| T24 | PASSED | cpu_unit_no_upstream |  |
| T25 | PASSED | cpu_unit_no_upstream |  |
| T26 | PASSED | cpu_unit_no_upstream |  |
| T27 | PASSED | cpu_unit_no_upstream |  |
| T28 | PASSED | cpu_unit_no_upstream |  |
| T29 | PASSED | cpu_unit_no_upstream |  |
| T30 | PASSED | cpu_unit_no_upstream |  |
| T31 | PASSED | cpu_unit_no_upstream |  |
| T32 | PASSED | cpu_unit_no_upstream |  |
| T33 | PASSED | cpu_virtual_delay_mock_planner |  |
| T34 | PASSED | cpu_virtual_delay_mock_planner |  |
| T35 | PASSED | cpu_virtual_delay_mock_planner |  |
| T36 | PASSED | cpu_virtual_delay_mock_planner |  |
| T37 | PASSED | cpu_virtual_delay_mock_planner |  |
| T38 | PASSED | cpu_virtual_delay_mock_planner |  |
| T39 | PASSED | cpu_virtual_delay_mock_planner |  |
| T40 | PASSED | cpu_virtual_delay_mock_planner |  |
| T41 | PASSED | cpu_virtual_delay_mock_planner |  |
| T42 | PASSED | cpu_virtual_delay_mock_planner |  |
| T43 | PASSED | cpu_virtual_delay_mock_planner |  |
| T44 | PASSED | cpu_wallclock_mock_planner |  |
| T45 | PASSED | cpu_virtual_delay_mock_planner |  |
| T46 | PASSED | cpu_virtual_delay_mock_planner |  |
| T47 | PASSED | cpu_virtual_delay_mock_planner |  |
| T48 | PASSED | cpu_virtual_delay_mock_planner |  |
| T49 | PASSED | cpu_virtual_delay_mock_planner |  |
| T50 | PASSED | cpu_virtual_delay_mock_planner |  |
| T51 | PASSED | cpu_virtual_delay_mock_planner |  |
| T52 | PASSED | cpu_virtual_delay_mock_planner |  |
| T53 | PASSED | cpu_virtual_delay_mock_planner |  |
| T54 | PASSED | cpu_virtual_delay_mock_planner |  |
| T55 | PASSED | cpu_virtual_delay_mock_planner |  |
| T56 | PASSED | cpu_virtual_delay_mock_planner |  |
| T57 | PASSED | cpu_virtual_delay_mock_planner |  |
| T58 | PASSED | cpu_virtual_delay_mock_planner |  |
| T59 | PASSED | cpu_virtual_delay_mock_planner |  |
| T60 | PASSED | cpu_toy_mock_features |  |
| T61 | SKIPPED | not_executed | Skipped: T61 requires the pinned RLinf B0 baseline, real data and weights / real-environment prerequisite: pinned RLinf  |
| T62 | SKIPPED | not_executed | Skipped: T62 requires pi0.5 + RLT + full_vjp with real weights; the CPU toy pipeline exercises the wiring but cannot sub |
| T63 | SKIPPED | not_executed | Skipped: T63 requires the real wall-clock B2 loop; a CPU wall-clock contract test with an explicitly labelled mock plann |
| T64 | SKIPPED | not_executed | Skipped: T64 requires the real learner running concurrently with the planner / real-environment prerequisite: real learn |
| T65 | PASSED | cpu_toy_mock_features |  |
| T66 | PASSED | cpu_toy_mock_features |  |
| T67 | PASSED | cpu_toy_mock_features |  |
| T68 | PASSED | cpu_virtual_delay_mock_planner |  |

PASSED rows were produced by a real CPU execution of this repository's tests.
SKIPPED rows require the real dataset/weights/GPU/physical robot and claim nothing.
