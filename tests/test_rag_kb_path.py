"""The knowledge base is found whatever directory the service was started from."""

import os

import rag


def test_a_relative_path_is_read_from_the_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = rag._resolve_kb_path("knowledge_base/EZZLAB.txt")
    assert os.path.isabs(resolved) and os.path.exists(resolved)


def test_a_bare_file_name_is_read_from_the_knowledge_base_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert rag._resolve_kb_path("EZZLAB.txt") == os.path.join(rag._PROJECT_ROOT, "knowledge_base", "EZZLAB.txt")


def test_a_missing_file_is_left_as_given():
    assert rag._resolve_kb_path("knowledge_base/nope.txt") == "knowledge_base/nope.txt"
