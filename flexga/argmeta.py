import typing as t
from enum import Enum
from abc import ABC, abstractmethod
import random

import numpy as np

from flexga.utils import is_one_of_types, is_two_tuple


class DType(Enum):
    CATEGORICAL = "categorical"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    FLOAT_VECTOR = "float-vector"


class ArgMeta(ABC):
    """
    Represents arguments that will be optimized via a
    standard binary genetic algorithm process.
    """

    @abstractmethod
    def mutate(self, x: t.Any) -> t.Any:
        """
        It is assumed the caller of this function has
        already determined a mutation needs to happen.
        """
        pass

    @abstractmethod
    def sample(self) -> t.Any:
        """
        Randomly samples a value of the argument. Used for
        initializing new GA populations.
        """
        pass


class RealArgMeta(ArgMeta):
    """
    Abstract class representing arguments that will be optimized via
    a real-encoded genetic algorithm process.
    """

    def __init__(self, bounds: t.Tuple, mutation_std) -> None:
        self.lbound = bounds[0]
        self.ubound = bounds[1]
        self.mutation_std = mutation_std

    @abstractmethod
    def crossover(
        self, x_1: t.Any, x_2: t.Any, x_1_fitness: float, x_2_fitness: float
    ) -> t.Tuple[t.Any, t.Any]:
        pass

    def enforce_bounds(self, x: t.Any) -> t.Any:
        ubounded = np.minimum(x, self.ubound)
        lbounded = np.maximum(ubounded, self.lbound)
        return lbounded


class CategoricalArgMeta(ArgMeta):
    def __init__(self, options: t.Iterable[t.Any]) -> None:
        self.options = set(options)

    def mutate(self, x: t.Any) -> t.Any:
        # Something random that is not x
        return random.choice(tuple(self.options - set(x)))

    def sample(self) -> t.Any:
        return random.choice(tuple(self.options))


class BoolArgMeta(ArgMeta):
    def mutate(self, x: bool) -> bool:
        return not x

    def sample(self) -> bool:
        return random.random() < 0.5


class IntArgMeta(RealArgMeta):
    def __init__(self, bounds: t.Tuple[int, int], mutation_std: int) -> None:
        if not is_two_tuple(bounds) or not all(
            is_one_of_types(val, (int, float)) for val in bounds
        ):
            raise ValueError(
                f"the bounds of an int arg must be of the format `(lb, ub)`."
            )
        super().__init__(bounds, mutation_std)

    def mutate(self, x: int) -> int:
        return x + int(round(random.gauss(0, self.mutation_std)))

    def crossover(
        self, x_1: int, x_2: int, x_1_fitness: float, x_2_fitness: float
    ) -> t.Tuple[int, int]:
        # child 1 is an average of the parents.
        c_1 = 0.5 * x_1 + 0.5 * x_2
        # child 2 is an extrapolation of the line formed by
        # x_1 and x_2. the direction of the extrapolation
        # is in the direction where objective values are estimated
        # by the extrapolation to improve.
        c_2 = 2 * x_2 - x_1 if x_2_fitness > x_1_fitness else 2 * x_1 - x_2
        return (
            self.enforce_bounds(int(round(c_1))),
            self.enforce_bounds(int(round(c_2))),
        )

    def sample(self) -> int:
        return random.randint(self.lbound, self.ubound)


class FloatArgMeta(RealArgMeta):
    def __init__(self, bounds: t.Tuple[float, float], mutation_std: float) -> None:
        if not is_two_tuple(bounds) or not all(
            is_one_of_types(val, (int, float)) for val in bounds
        ):
            raise ValueError(
                f"the bounds of a float arg must be of the format `(lb, ub)`."
            )
        super().__init__(bounds, mutation_std)

    def mutate(self, x: float) -> float:
        return x + random.gauss(0, self.mutation_std)

    def crossover(
        self, x_1: float, x_2: float, x_1_fitness: float, x_2_fitness: float
    ) -> t.Tuple[float, float]:
        # Same process as IntArgMeta without the rounding.
        c_1 = 0.5 * x_1 + 0.5 * x_2
        c_2 = 2 * x_2 - x_1 if x_2_fitness > x_1_fitness else 2 * x_1 - x_2
        return self.enforce_bounds(c_1), self.enforce_bounds(c_2)

    def sample(self) -> float:
        return random.uniform(self.lbound, self.ubound)


class FloatVectorArgMeta(RealArgMeta):
    def __init__(
        self, bounds: t.Tuple[np.ndarray, np.ndarray], mutation_std: float,
    ) -> None:
        if not is_two_tuple(bounds) or not all(
            isinstance(val, np.ndarray) and val.ndim == 1 for val in bounds
        ):
            raise ValueError(
                "the range of a float vector arg must be of the format "
                "`(np.ndarray, np.ndarray)`, where the arrays act as the "
                "bounds on the float vector. Each array must be 1D and both "
                "must have equal dimensionality."
            )
        super().__init__(bounds, mutation_std)
        assert self.lbound.size == self.ubound.size, "bounds have mismatched dimensions"

    def mutate(self, x: np.ndarray) -> np.ndarray:
        # Mutations are draw independently from a normal
        # distribution. The same mean and variance is used
        # for each dimension.
        return x + np.random.normal(0.0, self.mutation_std, x.size)

    def crossover(
        self, x_1: np.ndarray, x_2: np.ndarray, x_1_fitness: float, x_2_fitness: float
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        # Same process as IntArgMeta without the rounding.
        c_1 = 0.5 * x_1 + 0.5 * x_2
        c_2 = 2 * x_2 - x_1 if x_2_fitness > x_1_fitness else 2 * x_1 - x_2
        return self.enforce_bounds(c_1), self.enforce_bounds(c_2)

    def sample(self) -> np.ndarray:
        return np.random.random(self.lbound.size)
