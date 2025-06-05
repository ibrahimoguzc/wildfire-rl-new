# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Annotated, ClassVar, Protocol, TypeVar

import numpy as np
from annotated_types import Ge, Le, Lt
from pydantic import BaseModel
from pydantic_core import core_schema
from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

T = TypeVar("T")


Fraction = Annotated[float, Ge(0), Le(1)]
Percentage = Annotated[float, Ge(0), Le(100)]
HourOfDay = Annotated[float, Ge(0), Lt(24)]
AngleInDegree = Annotated[float, Ge(0), Lt(360)]


class PolymorphicBaseModel(BaseModel):
    """Base class for polymorphic models.

    This class is used to create polymorphic models. The class is
    initialized with a `polymorphic` flag, which is used to determine
    whether the model is polymorphic or not. If the model is
    polymorphic, the model will redirect the validation to a subclass.
    The `__identification_field__` field is used to identify the
    subclass that should be used to validate the model.

    Example::

        class Base(PolymorphicBaseModel, polymorphic=True):
            __identification_field__ = "type"


        class A(Base):
            type: str = "A"
            a: int


        class B(Base):
            type: str = "B"
            b: int


        assert Base.model_validate({"type": "A", "a": 1}) == A(a=1)
        assert Base.model_validate({"type": "B", "b": 2}) == B(b=2)
    """

    __polymorphic__: ClassVar[bool] = False
    __identification_field__: ClassVar[str]

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: type[Self],
        handler: Callable[[object], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        schema = handler(source)
        og_schema_ref = schema["ref"] + ":aux"
        return core_schema.no_info_before_validator_function(
            cls.__redirect_validation__, schema=schema, ref=og_schema_ref
        )

    @classmethod
    def __redirect_validation__(cls, value: object) -> Self:
        if not cls.__polymorphic__ or not isinstance(value, dict):
            return value

        field = cls.__identification_field__
        check_required_fields(field, value)
        identifier = value[field]
        for subclass in cls.__subclasses__():
            if subclass.model_fields[field].default == identifier:
                return subclass.model_validate(value)
        raise ValueError(
            f"Unknown {field!r} {identifier!r} in {cls.__name__!r}."
        )

    def __init_subclass__(
        cls, polymorphic: bool = False, **kwargs: object
    ) -> None:
        cls.__polymorphic__ = polymorphic
        super().__init_subclass__(**kwargs)
        if polymorphic and not hasattr(cls, "__identification_field__"):
            raise ValueError(
                f"Polymorphic model {cls.__name__!r} must define"
                f" __identification_field__."
            )


def check_field(
    field: str, dataset: dict | object, context: str | None = None
) -> None:
    """Check whether `field` is defined in `dataset`.

    `context`: input allows more information to be included in error in
        the case that the searched term is within a subdirectory.
    """
    # Use dir as hasattr does not work for properties.
    if not (field in dir(dataset) or field in dataset):
        if context is not None:
            raise ValueError(
                f"'{field}' could not be found, while checking for {context}"
                f", check validators of class parsing values for `{field}`."
            )
        raise ValueError(
            f"'{field}' could not be found, check validation errors"
        )


def check_required_fields(
    fields: str | Iterable[str],
    dataset: dict | object,
    subdirectory: str | None = None,
) -> None:
    """Check whether entries in `fields` are defined in `dataset`.

    subdirectory: if supplied, checks whether it exists, and `field` is
        searched for within subdirectory.
    """
    if subdirectory:
        # Check subdirectory exists
        check_field(subdirectory, dataset, fields)
        # If exists, access directory
        if isinstance(dataset, dict):
            data = dataset[subdirectory]
        else:
            data = getattr(dataset, subdirectory)
    else:
        data = dataset
    if type(fields) in [list, tuple]:
        # If multiple inputs passed, check individually
        for field in fields:
            check_field(field, data)
    else:
        check_field(fields, data)


class Orderable(Protocol):
    """Protocol for orderable objects."""

    def __lt__(self, other: object) -> bool: ...


def to_frozen_array(
    dtype: np.dtype[T], *, zero_shape: int | tuple[int, ...] = (0,)
) -> Callable[[Sequence[T]], np.ndarray[T]]:
    """Create a function that casts a sequence to a frozen array."""

    def converter(value: Sequence[T]) -> np.ndarray[T]:
        """Convert the sequence to an immutable array."""
        arr = np.asarray(value, dtype=dtype)
        if arr.size == 0:
            arr = arr.reshape(zero_shape)
        arr.setflags(write=False)
        return arr

    return converter


def is_sorted(iterable: Iterable[Orderable]) -> bool:
    """Check if an iterable is sorted."""
    return all(a <= b for a, b in pairwise(iterable))
