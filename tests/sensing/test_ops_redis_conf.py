from pathlib import Path

import pytest


@pytest.mark.unit
def test_redis_conf_has_required_settings():
    text = Path("ops/redis/redis.conf").read_text()
    assert "appendonly yes" in text
    assert "appendfsync everysec" in text
    assert "maxmemory-policy noeviction" in text
    assert "maxmemory 256mb" in text
    # RDB snapshots explicitly disabled — AOF is the source of durability.
    assert "save \"\"" in text


@pytest.mark.unit
def test_backup_script_is_executable_and_handles_both_stores():
    import stat
    path = Path("ops/backup.sh")
    text = path.read_text()
    assert ".backup" in text                     # SQLite
    assert "BGREWRITEAOF" in text                # Redis
    assert "appendonly.aof" in text              # the artifact being copied
    mode = path.stat().st_mode
    import sys
    if sys.platform != "win32":
        assert mode & stat.S_IXUSR
