from pathlib import Path
import pytest
from pi05_rlt_rtc.deployment.stage2_session import require_output_storage


def test_missing_removable_disk_fails_before_directory_creation(monkeypatch):
    monkeypatch.setattr(Path,'is_mount',lambda self:False)
    with pytest.raises(RuntimeError,match='未挂载'):
        require_output_storage(Path('/media/heinrich/SANDISK SSD/runs'))


def test_writable_local_output_is_verified_without_leftover_probe(tmp_path):
    root=tmp_path/'runs'
    require_output_storage(root)
    assert root.is_dir() and list(root.iterdir())==[]
