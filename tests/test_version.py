"""
Tests for cleaner.version
"""

from __future__ import annotations

import cleaner
from cleaner.version import VERSION, VERSION_INFO, get_version


class TestVersion:
    def test_version_is_string(self):
        assert isinstance(VERSION, str)

    def test_version_non_empty(self):
        assert len(VERSION) > 0

    def test_version_format(self):
        parts = VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_version_info_is_tuple(self):
        assert isinstance(VERSION_INFO, tuple)

    def test_version_info_length(self):
        assert len(VERSION_INFO) == 3

    def test_version_info_all_ints(self):
        for part in VERSION_INFO:
            assert isinstance(part, int)

    def test_version_info_matches_version(self):
        major, minor, patch = VERSION_INFO
        assert f"{major}.{minor}.{patch}" == VERSION

    def test_get_version_returns_string(self):
        result = get_version()
        assert isinstance(result, str)

    def test_get_version_equals_version(self):
        assert get_version() == VERSION

    def test_package_dunder_version(self):
        assert cleaner.__version__ == VERSION
