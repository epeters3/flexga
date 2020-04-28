import typing as t
import random
from copy import deepcopy

from flexga.argmeta import ArgMeta, RealArgMeta


class Arg:
    """
    Represents an arg and its meta data.
    """

    def __init__(self, value, meta: ArgMeta) -> None:
        self.value = value  # the arg value
        self.meta = meta  # metadata about the arg

    def mutate(self) -> None:
        """
        It is assumed the caller of this function has
        already determined a mutation needs to happen.
        """
        self.value = self.meta.mutate(self.value)


class Genome:
    def __init__(
        self, args: t.Sequence[Arg], kwargs: t.Dict[str, Arg],
    ):
        self.args = args
        self.kwargs = kwargs
        self.fitness: float = None  # type: ignore

    def mutate(self, p: float) -> None:
        """
        Mutates each attribute in self with probability `p`.
        """
        for arg in self.args:
            if random.random() < p:
                arg.mutate()

    def get_arg_vals(self) -> t.Tuple[t.Sequence, t.Dict]:
        return (
            [a.value for a in self.args],
            {name: a.value for name, a in self.kwargs.items()},
        )

    def evaluate_fitness(self, objective: t.Callable) -> None:
        args, kwargs = self.get_arg_vals()
        self.fitness = objective(*args, **kwargs)

    @staticmethod
    def crossover(a: "Genome", b: "Genome") -> t.Tuple["Genome", "Genome"]:
        """
        Crosses over `a` and `b`, returning their two children. Uses a
        version of multi-point crossover.
        """
        # This determines how evenly the parent's arguments are divided
        # among the children. If it's very low, child a will get most of
        # parent a's genes (arguments), and child b will get most of
        # parent b's. If it's = 0.5, each child should get about half
        # of each parent's genes.
        split_ratio = random.random()

        childa_args = []
        childa_kwargs = {}
        childb_args = []
        childb_kwargs = {}

        for a_arg, b_arg in zip(a.args, b.args):
            if isinstance(a_arg.meta, RealArgMeta):
                # Do real-valued crossover
                childa_value, childb_value = a_arg.meta.crossover(
                    a_arg.value, b_arg.value, a.fitness, b.fitness
                )
                childa_args.append(Arg(childa_value, a_arg.meta))
                childb_args.append(Arg(childb_value, b_arg.meta))
            else:
                # Do binary crossover
                if random.random() < split_ratio:
                    childa_args.append(deepcopy(a_arg))
                    childb_args.append(deepcopy(b_arg))
                else:
                    childa_args.append(deepcopy(b_arg))
                    childb_args.append(deepcopy(a_arg))

        for name in a.kwargs:
            a_arg = a.kwargs[name]
            b_arg = b.kwargs[name]
            if isinstance(a_arg.meta, RealArgMeta):
                # Do real-valued crossover
                childa_value, childb_value = a_arg.meta.crossover(
                    a_arg.value, b_arg.value, a.fitness, b.fitness
                )
                childa_kwargs[name] = Arg(childa_value, a_arg.meta)
                childb_kwargs[name] = Arg(childb_value, b_arg.meta)
            else:
                # Do binary crossover
                if random.random() < split_ratio:
                    childa_kwargs[name] = deepcopy(a_arg)
                    childb_kwargs[name] = deepcopy(b_arg)
                else:
                    childa_kwargs[name] = deepcopy(b_arg)
                    childb_kwargs[name] = deepcopy(a_arg)

        return Genome(childa_args, childa_kwargs), Genome(childb_args, childb_kwargs)

    @classmethod
    def sample(
        cls, argsmeta: t.Sequence[ArgMeta], kwargsmeta: t.Dict[str, ArgMeta]
    ) -> "Genome":
        """
        Samples a randomly initialized `Genome` instance, initialized according to
        the specs declared in `argsmeta` and `kwargsmeta`.
        """
        args: t.List[Arg] = [Arg(argmeta.sample(), argmeta) for argmeta in argsmeta]
        kwargs: t.Dict[str, Arg] = {
            name: Arg(argmeta.sample(), argmeta) for name, argmeta in kwargsmeta.items()
        }
        return cls(args, kwargs)

    @classmethod
    def sample_n(
        cls, argsmeta: t.Sequence[ArgMeta], kwargsmeta: t.Dict[str, ArgMeta], n: int
    ) -> t.List["Genome"]:
        """
        Samples `n` randomly initialized `Genome` instances.
        """
        return [cls.sample(argsmeta, kwargsmeta) for _ in range(n)]
