from __future__ import annotations

import shutil

from fraud_detection.components.model_pusher import _copy_promoted_file


def test_copy_promoted_file_falls_back_to_copyfile_on_permission_error(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    source.write_text("promotion payload", encoding="utf-8")

    def fake_copy2(src, dst):
        raise PermissionError("metadata write not permitted")

    fallback_calls = []

    def fake_copyfile(src, dst):
        fallback_calls.append((src, dst))
        return shutil.copyfile.__wrapped__(src, dst)  # type: ignore[attr-defined]

    original_copyfile = shutil.copyfile

    def wrapped_copyfile(src, dst):
        fallback_calls.append((src, dst))
        return original_copyfile(src, dst)

    monkeypatch.setattr(shutil, "copy2", fake_copy2)
    monkeypatch.setattr(shutil, "copyfile", wrapped_copyfile)

    _copy_promoted_file(source, target)

    assert fallback_calls == [(source, target)]
    assert target.read_text(encoding="utf-8") == "promotion payload"