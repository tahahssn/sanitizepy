"""
sanitizepy.cleaning.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Registry responsible for mapping operation names to their
:class:`CleaningOperation` classes.

The registry mirrors the ABC-plus-registry pattern used by
:class:`sanitizepy.rules.registry.RuleRegistry`, but keys entries by the
operation ``name`` so plans can be reconstructed from serialized parameters
via a name-to-class lookup.
"""

from __future__ import annotations

from collections.abc import Iterator

from .base import CleaningOperation


class OperationRegistry:
    """
    Registry mapping operation names to :class:`CleaningOperation` classes.

    Operations are uniquely identified by their ``name`` attribute.
    """

    def __init__(self) -> None:
        self._operations: dict[str, type[CleaningOperation]] = {}

    def register(self, name: str, operation_cls: type[CleaningOperation]) -> None:
        """
        Register an operation class under the given name.

        Raises
        ------
        ValueError
            If an operation is already registered under ``name``.
        """
        if name in self._operations:
            raise ValueError(f"Operation '{name}' is already registered.")

        self._operations[name] = operation_cls

    def unregister(self, name: str) -> None:
        """
        Remove a registered operation.
        """
        del self._operations[name]

    def get(self, name: str) -> type[CleaningOperation]:
        """
        Return a registered operation class.

        Raises
        ------
        KeyError
            If no operation is registered under ``name``.
        """
        return self._operations[name]

    def contains(self, name: str) -> bool:
        """
        Check whether an operation is registered under ``name``.
        """
        return name in self._operations

    def clear(self) -> None:
        """
        Remove every registered operation.
        """
        self._operations.clear()

    def values(self) -> tuple[type[CleaningOperation], ...]:
        """
        Return all registered operation classes.
        """
        return tuple(self._operations.values())

    def names(self) -> tuple[str, ...]:
        """
        Return registered operation names.
        """
        return tuple(self._operations.keys())

    def items(self) -> tuple[tuple[str, type[CleaningOperation]], ...]:
        """
        Return (name, operation class) pairs.
        """
        return tuple(self._operations.items())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._operations

    def __len__(self) -> int:
        return len(self._operations)

    def __iter__(self) -> Iterator[type[CleaningOperation]]:
        return iter(self._operations.values())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(operations={len(self)})"


registry = OperationRegistry()
"""Module-level registry instance with the built-in operations registered."""


def register_builtin_operations(target: OperationRegistry) -> None:
    """
    Register the built-in cleaning operations on ``target`` by their ``name``.
    """
    from .encoding import EncodingRepairOperation
    from .missing_tokens import MissingTokenOperation
    from .near_duplicates import NearDuplicateRemovalOperation
    from .operations import (
        DropColumns,
        DropDuplicates,
        DropMissingColumns,
        DropMissingRows,
        FillMissing,
    )
    from .text_normalization import TextNormalizationOperation
    from .type_coercion import TypeCoercionOperation

    builtins: tuple[type[CleaningOperation], ...] = (
        DropMissingRows,
        DropMissingColumns,
        FillMissing,
        DropDuplicates,
        DropColumns,
        TypeCoercionOperation,
        MissingTokenOperation,
        TextNormalizationOperation,
        EncodingRepairOperation,
        NearDuplicateRemovalOperation,
    )

    for operation_cls in builtins:
        if not target.contains(operation_cls.name):
            target.register(operation_cls.name, operation_cls)


register_builtin_operations(registry)


__all__ = [
    "OperationRegistry",
    "registry",
    "register_builtin_operations",
]
