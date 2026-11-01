"""Two explicit, reproducible registration edits to the pinned RLinf checkout."""
import subprocess
from pathlib import Path

from pi05_rlt_rtc.adapters.source_audit import PINNED_RLINF_COMMIT
from pi05_rlt_rtc.contracts.errors import ContractError

REGISTRY = "rlinf/models/embodiment/openpi/dataconfig/__init__.py"
LOADER = "rlinf/data/datasets/openpi/official_sft_data_loader.py"


def expected_sources(root):
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if sha != PINNED_RLINF_COMMIT:
        raise ContractError("physical profile registration requires the pinned RLinf revision")
    expected = {}
    changes = {
        REGISTRY: ('if len({config.name for config in _CONFIGS}) != len(_CONFIGS):',
            'from pi05_rlt_rtc.adapters.panda_policy import panda_train_config\n'
            '_CONFIGS.append(panda_train_config())\n\n'
            'if len({config.name for config in _CONFIGS}) != len(_CONFIGS):'),
        LOADER: ('    del rank\n',
            '    if cfg.actor.model.openpi.config_name == "pi05_rlt_panda_robotiq_joint":\n'
            '        from pi05_rlt_rtc.data.physical_sft import build_sft_dataloader\n'
            '        return build_sft_dataloader(cfg, world_size, rank, data_paths, eval_dataset)\n'
            '    del rank\n')}
    for name, (before, after) in changes.items():
        original = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{name}"], text=True)
        if original.count(before) != 1:
            raise ContractError(f"unexpected registration anchor in {name}")
        expected[name] = (original, original.replace(before, after))
    return expected


def verify_profile_checkout(root):
    expected = expected_sources(root)
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", "rlinf"], text=True)
    for line in status.splitlines():
        name = line[3:]
        if name not in expected or (Path(root) / name).read_text() != expected[name][1]:
            raise ContractError(f"unverified RLinf source modification: {name}")


def install(root):
    root = Path(root).expanduser().resolve()
    expected = expected_sources(root)
    verify_profile_checkout(root)
    for name, versions in expected.items():
        if (root / name).read_text() not in versions:
            raise ContractError(f"preserving unrelated edits in {name}; use an isolated pinned checkout")
    for name, (_, patched) in expected.items():
        (root / name).write_text(patched)
    verify_profile_checkout(root)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-root", required=True)
    args = parser.parse_args()
    install(args.rlinf_root)
    print("Registered the physical Panda model profile and LeRobot loader.")
