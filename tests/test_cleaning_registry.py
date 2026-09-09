"""
Tests for sanitizepy.cleaning.registry (OperationRegistry)

Coverage:
- register / get round-trip
- unknown-name lookup raises KeyError
- duplicate registration raises ValueError
- unregister removes an entry
- contains / __contains__ / __len__ / __iter__
- names() / values() / items()
- clear() empties the registry
- module-level ``registry`` instance has built-in operations registered
- register_builtin_operations is idempotent (re-registration skipped)

Requirements: 13.1, 14.3
"""

from __future__ import annotations

import pandas as pd
import pytest

from sanitizepy.cleaning.base import CleaningOperation
from sanitizepy.cleaning.operations import (
    DropColumns,
    DropDuplicates,
    DropMissingColumns,
    DropMissingRows,
    FillMissing,
)
from sanitizepy.cleaning.registry import (
    OperationRegistry,
    register_builtin_operations,
    registry,
)
from sanitizepy.cleaning.type_coercion import TypeCoercionOperation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BUILTIN_NAMES: tuple[str, ...] = (
    DropMissingRows.name,
    DropMissingColumns.name,
    FillMissing.name,
    DropDuplicates.name,
    DropColumns.name,
    TypeCoercionOperation.name,
)


class _Alpha(CleaningOperation):
    name = "alpha"

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)
        return dataframe.copy()


class _Beta(CleaningOperation):
    name = "beta"

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)
        return dataframe.copy()


# ---------------------------------------------------------------------------
# Fresh-registry fixture so tests are isolated from the module-level instance
# ---------------------------------------------------------------------------


@pytest.fixture()
def empty_registry() -> OperationRegistry:
    """A brand-new, empty OperationRegistry."""
    return OperationRegistry()


@pytest.fixture()
def seeded_registry(empty_registry: OperationRegistry) -> OperationRegistry:
    """Registry pre-loaded with _Alpha and _Beta."""
    empty_registry.register(_Alpha.name, _Alpha)
    empty_registry.register(_Beta.name, _Beta)
    return empty_registry


# ===========================================================================
# Register / get round-trip
# ===========================================================================


class TestRegisterGetRoundTrip:
    def test_get_returns_registered_class(self, empty_registry: OperationRegistry):
        empty_registry.register("alpha", _Alpha)
        assert empty_registry.get("alpha") is _Alpha

    def test_get_after_two_registrations(self, empty_registry: OperationRegistry):
        empty_registry.register("alpha", _Alpha)
        empty_registry.register("beta", _Beta)
        assert empty_registry.get("alpha") is _Alpha
        assert empty_registry.get("beta") is _Beta

    def test_registered_class_is_instantiable(self, empty_registry: OperationRegistry):
        empty_registry.register("alpha", _Alpha)
        retrieved_cls = empty_registry.get("alpha")
        instance = retrieved_cls()
        assert isinstance(instance, CleaningOperation)

    def test_register_uses_caller_key_not_class_name(
        self, empty_registry: OperationRegistry
    ):
        # Keys are caller-controlled; the operation's own `.name` attribute is
        # separate from the registry key even if they happen to differ.
        empty_registry.register("custom_key", _Alpha)
        assert empty_registry.get("custom_key") is _Alpha
        with pytest.raises(KeyError):
            empty_registry.get(_Alpha.name)


# ===========================================================================
# Unknown-name lookup raises KeyError
# ===========================================================================


class TestUnknownLookup:
    def test_get_unknown_name_raises_key_error(self, empty_registry: OperationRegistry):
        with pytest.raises(KeyError):
            empty_registry.get("nonexistent")

    def test_get_unknown_name_in_nonempty_registry_raises(
        self, seeded_registry: OperationRegistry
    ):
        with pytest.raises(KeyError):
            seeded_registry.get("nonexistent")

    def test_get_empty_string_raises(self, empty_registry: OperationRegistry):
        with pytest.raises(KeyError):
            empty_registry.get("")

    def test_get_after_unregister_raises(self, seeded_registry: OperationRegistry):
        seeded_registry.unregister("alpha")
        with pytest.raises(KeyError):
            seeded_registry.get("alpha")


# ===========================================================================
# Duplicate registration raises ValueError
# ===========================================================================


class TestDuplicateRegistration:
    def test_duplicate_registration_raises_value_error(
        self, empty_registry: OperationRegistry
    ):
        empty_registry.register("alpha", _Alpha)
        with pytest.raises(ValueError, match="already registered"):
            empty_registry.register("alpha", _Alpha)

    def test_duplicate_registration_with_different_class_raises(
        self, empty_registry: OperationRegistry
    ):
        empty_registry.register("slot", _Alpha)
        # Trying to claim the same key with a different class must also raise.
        with pytest.raises(ValueError, match="already registered"):
            empty_registry.register("slot", _Beta)

    def test_original_remains_after_duplicate_attempt(
        self, empty_registry: OperationRegistry
    ):
        empty_registry.register("alpha", _Alpha)
        with pytest.raises(ValueError):
            empty_registry.register("alpha", _Beta)
        # The registry still holds the original mapping.
        assert empty_registry.get("alpha") is _Alpha

    def test_error_message_includes_operation_name(
        self, empty_registry: OperationRegistry
    ):
        empty_registry.register("my_op", _Alpha)
        with pytest.raises(ValueError, match="my_op"):
            empty_registry.register("my_op", _Alpha)


# ===========================================================================
# Membership / introspection helpers
# ===========================================================================


class TestMembershipAndIntrospection:
    def test_contains_method_true(self, seeded_registry: OperationRegistry):
        assert seeded_registry.contains("alpha") is True

    def test_contains_method_false(self, seeded_registry: OperationRegistry):
        assert seeded_registry.contains("nonexistent") is False

    def test_dunder_contains_true(self, seeded_registry: OperationRegistry):
        assert "alpha" in seeded_registry

    def test_dunder_contains_false(self, seeded_registry: OperationRegistry):
        assert "nonexistent" not in seeded_registry

    def test_len_empty(self, empty_registry: OperationRegistry):
        assert len(empty_registry) == 0

    def test_len_after_registration(self, empty_registry: OperationRegistry):
        empty_registry.register("alpha", _Alpha)
        assert len(empty_registry) == 1
        empty_registry.register("beta", _Beta)
        assert len(empty_registry) == 2

    def test_names_returns_all_keys(self, seeded_registry: OperationRegistry):
        assert set(seeded_registry.names()) == {"alpha", "beta"}

    def test_values_returns_all_classes(self, seeded_registry: OperationRegistry):
        assert set(seeded_registry.values()) == {_Alpha, _Beta}

    def test_items_returns_pairs(self, seeded_registry: OperationRegistry):
        pairs = dict(seeded_registry.items())
        assert pairs == {"alpha": _Alpha, "beta": _Beta}

    def test_iter_yields_classes(self, seeded_registry: OperationRegistry):
        classes = list(seeded_registry)
        assert set(classes) == {_Alpha, _Beta}

    def test_repr_includes_count(self, seeded_registry: OperationRegistry):
        r = repr(seeded_registry)
        assert "2" in r


# ===========================================================================
# Unregister
# ===========================================================================


class TestUnregister:
    def test_unregister_removes_entry(self, seeded_registry: OperationRegistry):
        seeded_registry.unregister("alpha")
        assert not seeded_registry.contains("alpha")

    def test_len_decreases_after_unregister(self, seeded_registry: OperationRegistry):
        seeded_registry.unregister("alpha")
        assert len(seeded_registry) == 1

    def test_unregister_missing_key_raises(self, empty_registry: OperationRegistry):
        with pytest.raises(KeyError):
            empty_registry.unregister("nonexistent")

    def test_can_re_register_after_unregister(self, empty_registry: OperationRegistry):
        empty_registry.register("alpha", _Alpha)
        empty_registry.unregister("alpha")
        # After unregistering, the same key must be freely re-registerable.
        empty_registry.register("alpha", _Beta)
        assert empty_registry.get("alpha") is _Beta


# ===========================================================================
# clear()
# ===========================================================================


class TestClear:
    def test_clear_empties_registry(self, seeded_registry: OperationRegistry):
        seeded_registry.clear()
        assert len(seeded_registry) == 0

    def test_clear_on_empty_is_safe(self, empty_registry: OperationRegistry):
        empty_registry.clear()  # must not raise
        assert len(empty_registry) == 0

    def test_can_register_after_clear(self, seeded_registry: OperationRegistry):
        seeded_registry.clear()
        seeded_registry.register("alpha", _Alpha)
        assert seeded_registry.get("alpha") is _Alpha


# ===========================================================================
# Module-level ``registry`` has built-in operations registered
# ===========================================================================


class TestModuleLevelRegistry:
    @pytest.mark.parametrize("builtin_name", BUILTIN_NAMES)
    def test_builtin_operation_is_present(self, builtin_name: str):
        assert registry.contains(builtin_name), (
            f"Built-in operation '{builtin_name}' is missing from the "
            "module-level registry."
        )

    @pytest.mark.parametrize(
        ("builtin_name", "expected_cls"),
        [
            (DropMissingRows.name, DropMissingRows),
            (DropMissingColumns.name, DropMissingColumns),
            (FillMissing.name, FillMissing),
            (DropDuplicates.name, DropDuplicates),
            (DropColumns.name, DropColumns),
            (TypeCoercionOperation.name, TypeCoercionOperation),
        ],
    )
    def test_builtin_maps_to_correct_class(
        self, builtin_name: str, expected_cls: type[CleaningOperation]
    ):
        assert registry.get(builtin_name) is expected_cls

    def test_registry_has_at_least_all_builtins(self):
        assert len(registry) >= len(BUILTIN_NAMES)


# ===========================================================================
# register_builtin_operations is idempotent
# ===========================================================================


class TestRegisterBuiltinOperationsIdempotency:
    def test_calling_twice_does_not_raise(self, empty_registry: OperationRegistry):
        register_builtin_operations(empty_registry)
        # A second call must skip already-registered names without raising.
        register_builtin_operations(empty_registry)

    def test_count_unchanged_after_second_call(self, empty_registry: OperationRegistry):
        register_builtin_operations(empty_registry)
        count_after_first = len(empty_registry)
        register_builtin_operations(empty_registry)
        assert len(empty_registry) == count_after_first

    def test_all_builtins_present_after_idempotent_call(
        self, empty_registry: OperationRegistry
    ):
        register_builtin_operations(empty_registry)
        register_builtin_operations(empty_registry)
        for name in BUILTIN_NAMES:
            assert empty_registry.contains(name)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_register_many_operations(self, empty_registry: OperationRegistry):
        """Registry must handle a large number of entries without issue."""

        ops: list[type[CleaningOperation]] = []
        for i in range(50):

            class _DynamicOp(CleaningOperation):
                name = f"op_{i}"

                def apply(self, df: pd.DataFrame) -> pd.DataFrame:
                    return df.copy()

            _DynamicOp.name = f"dynamic_{i}"
            ops.append(_DynamicOp)
            empty_registry.register(_DynamicOp.name, _DynamicOp)

        assert len(empty_registry) == 50
        for op_cls in ops:
            assert empty_registry.get(op_cls.name) is op_cls

    def test_non_string_key_not_in_registry(self, seeded_registry: OperationRegistry):
        # ``__contains__`` accepts ``object`` and must return False for
        # non-string arguments rather than raising.
        assert 123 not in seeded_registry
        assert None not in seeded_registry

    def test_register_same_class_under_different_names(
        self, empty_registry: OperationRegistry
    ):
        """The same class can be registered under multiple distinct keys."""
        empty_registry.register("name_a", _Alpha)
        empty_registry.register("name_b", _Alpha)
        assert empty_registry.get("name_a") is _Alpha
        assert empty_registry.get("name_b") is _Alpha
        assert len(empty_registry) == 2
