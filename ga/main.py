import typing as t
import random
from copy import deepcopy

from ga.argmeta import ArgMeta, RealArgMeta
from ga.utils import grouper, shuffle


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


class GAPopulation:
    def __init__(
        self,
        argsmeta: t.Sequence[ArgMeta],
        kwargsmeta: t.Dict[str, ArgMeta],
        size: int,
        initialize_randomly: bool = False,
    ):
        """
        Creates a population. If `initialize_randomly == True`, the population
        will be filled with genomes randomly sampled within the bounds expressed
        in `argsmeta` and `kwargsmeta`. If `False`, the population members
        (`Genome` instances) will needed to be added to the population's
        `members` list by hand.
        """
        self.argsmeta = argsmeta
        self.kwargsmeta = kwargsmeta
        self.size = size
        if initialize_randomly:
            self.members = Genome.sample_n(self.argsmeta, self.kwargsmeta, self.size)
        else:
            self.members = []
        self.best_fitness: float = None  # type: ignore
        self.best_genome: Genome = None  # type: ignore

    def evaluate_fitness(self, objective: t.Callable) -> None:
        for genome in self.members:
            genome.evaluate_fitness(objective)
            if self.best_fitness is None or genome.fitness > self.best_fitness:
                # We've found a new best
                self.best_fitness = genome.fitness
                self.best_genome = genome

    def do_selection(self) -> t.Iterable[t.Tuple[int, int]]:
        """
        Select the parent couples to mate using tournament
        selection. Returns an iterable of parent index pairs.
        """
        # Tournament selection
        # Must be even number for tournament selection to work nicely.
        assert self.size % 2 == 0
        left_parents = self._do_half_selection()
        right_parents = self._do_half_selection()
        # Do a little matchmaking
        return zip(left_parents, right_parents)

    def crossover(self, parent_pairs: t.Iterable[t.Tuple[int, int]]) -> "GAPopulation":
        """
        Breed the parents to create a new population.
        """
        new_generation = GAPopulation(self.argsmeta, self.kwargsmeta, self.size)

        for a_i, b_i in parent_pairs:
            parent_a = self.members[a_i]
            parent_b = self.members[b_i]
            child_a, child_b = Genome.crossover(parent_a, parent_b)
            new_generation.members += [child_a, child_b]

        # Elitism - keep the best member of the population.
        new_generation.best_fitness = self.best_fitness
        new_generation.best_genome = self.best_genome
        return new_generation

    def mutate(self, p: float) -> None:
        for genome in self.members:
            genome.mutate(p)

    def _do_half_selection(self) -> t.Sequence:
        candidate_indices = shuffle(tuple(range(self.size)))
        parents = []
        for cand_a, cand_b in grouper(candidate_indices, 2):
            if self.members[cand_a].fitness > self.members[cand_b].fitness:
                parents.append(cand_a)
            else:
                parents.append(cand_b)
        return parents


def ga(
    fun: t.Callable,
    *,
    argsmeta: t.Sequence[ArgMeta] = None,
    kwargsmeta: t.Dict[str, ArgMeta] = None,
    iters: int,
    population_size: int = None,
    mutation_prob: float = 0.02,
    verbose: bool = False,
) -> t.Tuple[float, t.Sequence, t.Dict[str, t.Any]]:
    """
    Uses a genetic algorithm to minimize the output of `fun`.

    Parameters
    ----------
    fun:
        Should return a single value.
    argsmeta:
        A list of metadata about each positional argument in `fun`'s function
        signature. Each metadata object should be an instance of `ga.argmeta.ArgMeta`.
    kwargsmeta:
        A mapping of the key-word arg names in `fun`'s function signature to
        metadata about each of those args.
    iters:
        The number of optimization iterations (i.e. generations) to complete.
    population_size:
        The size of the population to maintain. If `None`, an order of magnitude
        larger than the number of arguments `fun` takes will be used. Note that
        if one or more of the arguments to `fun` are vectors, this value should
        be supplied manually.
    mutation_prob:
        The probability with which to mutate genes in new child genomes.
    verbose:
        Whether to print a status message each iteration.
    
    Returns
    -------
    fopt:
        The optimal output for `fun` found by the optimizer.
    args_opt:
        The positional arguments given to `fun` that yielded `fopt`.
    kwargs_opt:
        The key-word arguments given to `fun` that yielded `fopt`.
    """
    if argsmeta is None and kwargsmeta is None:
        raise ValueError(
            "no annotations provided for `fun`'s arguments: must populate "
            "`argsmeta` and/or `kwargsmeta`, depending on your `fun`'s "
            "function signature."
        )
    if argsmeta is None:
        argsmeta = []
    if kwargsmeta is None:
        kwargsmeta = {}
    if population_size is None:
        population_size = (len(argsmeta) + len(kwargsmeta)) * 10

    population = GAPopulation(argsmeta, kwargsmeta, population_size, True)
    for i in range(iters):
        # Compute objective for each genome in population
        population.evaluate_fitness(fun)
        if verbose:
            print(f"iter {i+1} => fopt: {population.best_fitness:6.6f}")
        # Choose points from population for the mating pool
        parent_pairs = population.do_selection()
        # Create a new population from the mating pool
        population = population.crossover(parent_pairs)
        # Randomly mutate some genomes in the population
        population.mutate(mutation_prob)
    # Return "optimum" (best result found), and the arguments to `fun`
    # used to find it.
    args_opt, kwargs_opt = population.best_genome.get_arg_vals()
    if verbose:
        print("fopt:", population.best_fitness)
        print("optimal args:", args_opt, kwargs_opt)
    return population.best_fitness, args_opt, kwargs_opt
