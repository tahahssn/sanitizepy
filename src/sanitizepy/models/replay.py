"""
Replayable plan models.

A :class:`ReplayablePlan` is a frozen, JSON-serializable snapshot of a
cleaning plan. It captures everything required to deterministically
reconstruct and re-run the exact sequence of cleaning operations that
produced a cleaned dataset:

* ``version`` -- the library version that produced the plan, so a
  replay can detect a version mismatch.
* ``operations`` -- an ordered sequence of :class:`ReplayOperation`
  entries. Each entry records the operation ``name`` (its registry key)
  and the JSON-serializable reconstruction ``parameters`` (typically the
  operation's ``describe()`` payload minus the redundant ``name`` key).
* ``configuration`` -- the JSON-serializable engine/config snapshot the
  plan was executed under.
* ``seeds`` -- deterministic seeds recorded per operation (keyed by
  operation name) for any operation that involves randomness.

The model carries no reconstruction logic itself: turning a
``ReplayablePlan`` back into live :class:`CleaningOperation` instances via
the :class:`~sanitizepy.cleaning.registry.OperationRegistry` is handled by
the cleaning subsystem. This module only owns the immutable data shape and
the JSON (de)serialization boundary.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field, ValidationError

from sanitizepy.exceptions import DeserializationError, SerializationError
from sanitizepy.models.base import BaseCleanerModel
from sanitizepy.version import VERSION


def _assert_json_serializable(value: Any, *, context: str) -> None:
    """
    Raise :class:`SerializationError` if ``value`` is not JSON-serializable.
    """
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise SerializationError(
            f"Non-JSON-serializable value in {context}: {exc}"
        ) from exc


class ReplayOperation(BaseCleanerModel):
    """
    A single operation entry within a :class:`ReplayablePlan`.

    Stores the deterministic reconstruction parameters for one cleaning
    operation. ``name`` is the operation's registry key (its ``name``
    attribute) and ``parameters`` is the JSON-serializable payload used to
    re-instantiate the operation.
    """

    name: str = Field(
        min_length=1,
        description="Registry key of the operation (its ``name``).",
    )

    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-serializable reconstruction parameters.",
    )


class ReplayablePlan(BaseCleanerModel):
    """
    A frozen, JSON-serializable snapshot of a cleaning plan.

    A ``ReplayablePlan`` records the ordered operations, the configuration
    they ran under, and any per-operation seeds, so the exact cleaning
    sequence can be reconstructed and replayed deterministically.
    """

    version: str = Field(
        default=VERSION,
        min_length=1,
        description="Library version that produced the plan.",
    )

    operations: tuple[ReplayOperation, ...] = Field(
        default_factory=tuple,
        description="Ordered operations required to replay the plan.",
    )

    configuration: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-serializable configuration snapshot.",
    )

    seeds: dict[str, int] = Field(
        default_factory=dict,
        description="Deterministic seeds keyed by operation name.",
    )

    def to_json(self) -> str:
        """
        Serialize the plan to a JSON string.

        Returns
        -------
        str
            A JSON document round-trippable via :meth:`from_json`.

        Raises
        ------
        SerializationError
            If any operation parameter or configuration value is not
            JSON-serializable.
        """
        for operation in self.operations:
            _assert_json_serializable(
                operation.parameters,
                context=f"operation '{operation.name}' parameters",
            )
        _assert_json_serializable(self.configuration, context="configuration")

        try:
            return self.model_dump_json()
        except (TypeError, ValueError) as exc:
            raise SerializationError(
                f"Failed to serialize replayable plan: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, data: str) -> ReplayablePlan:
        """
        Reconstruct a :class:`ReplayablePlan` from a JSON string.

        Parameters
        ----------
        data:
            A JSON document produced by :meth:`to_json`.

        Returns
        -------
        ReplayablePlan
            The reconstructed plan.

        Raises
        ------
        DeserializationError
            If ``data`` is not valid JSON or does not match the plan
            schema (for example an unknown or malformed operation entry).
        """
        try:
            return cls.model_validate_json(data)
        except ValidationError as exc:
            raise DeserializationError(
                f"Malformed replayable plan payload: {exc}"
            ) from exc


__all__ = [
    "ReplayOperation",
    "ReplayablePlan",
]
