import typing as t
import random
from copy import deepcopy

from flexga.argmeta import ArgMeta, RealArgMeta
from flexga.utils import grouper, shuffle, conditional_timeout, EvaluationTimeoutError


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
        self.best_fitness = -float("inf")
        self.best_genome: Genome = None  # type: ignore

    def evaluate_fitness(self, objective: t.Callable) -> None:
        for genome in self.members:
            genome.evaluate_fitness(objective)
            if genome.fitness > self.best_fitness:
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


def flexga(
    fun: t.Callable,
    *,
    argsmeta: t.Sequence[ArgMeta] = None,
    kwargsmeta: t.Dict[str, ArgMeta] = None,
    iters: t.Optional[int] = None,
    time: t.Optional[int] = None,
    patience: t.Optional[int] = 20,
    patience_tolerance: float = 0.0,
    population_size: int = None,
    mutation_prob: float = 0.02,
    print_every: t.Optional[int] = None,
    callback: t.Callable = None,
) -> t.Tuple[float, t.Sequence, t.Dict[str, t.Any]]:
    """
    Uses a genetic algorithm to maximize the output of `fun`.

    Parameters
    ----------
    fun:
        Should return a single value.
    argsmeta:
        A list of metadata about each positional argument in `fun`'s function
        signature. Each metadata object should be an instance of
        `flexga.argmeta.ArgMeta`.
    kwargsmeta:
        A mapping of the key-word arg names in `fun`'s function signature to
        metadata about each of those args.
    iters:
        If supplied, the maximum number of optimization iterations (i.e. generations)
        will not exceed `iters`.
    time:
        If supplied, optimization will quit after `time` seconds, returning the best
        solution it could find in that time.
    patience:
        If supplied, and the current best optimum is not improved upon by at least
        `patience_tolerance` for `patience` generations, the optimizer will exit.
    patience_tolerance:
        See `patience`.
    population_size:
        The size of the population to maintain. If `None`, an order of magnitude
        larger than the number of arguments `fun` takes will be used. Note that
        if one or more of the arguments to `fun` are vectors, this value should
        be supplied manually.
    mutation_prob:
        The probability with which to mutate genes in new child genomes.
    print_every:
        If supplied, a status message will be printed every `print_every` generations.
    callback:
        An optional callback function that will be called after every generation. The
        function signature should be `callback(state: dict) -> bool`, where `state` is
        a dictionary containing data about the current state of the optimizer, namely
        these fields:
            - "fopt": The optimal output for `fun` found by the optimizer so far
            - "args_opt": The positional arguments given to `fun` that yielded `fopt`
            - "kwargs_opt": The key-word arguments given to `fun` that yielded `fopt`
            - "nit": The number of generations/iterations completed so far.
        
        . If `callback` returns `True`, the optimization will end prematurely.
    
    Returns
    -------
    fopt:
        The optimal output for `fun` found by the optimizer.
    args_opt:
        The positional arguments given to `fun` that yielded `fopt`.
    kwargs_opt:
        The key-word arguments given to `fun` that yielded `fopt`.
    """
    # Validate inputs
    if argsmeta is None and kwargsmeta is None:
        raise ValueError(
            "no annotations provided for `fun`'s arguments: must populate "
            "`argsmeta` and/or `kwargsmeta`, depending on your `fun`'s "
            "function signature."
        )
    if patience is None and iters is None:
        raise ValueError(
            "must supply a value for either the `patience` or `iters` argument."
        )

    # Provide default values
    if argsmeta is None:
        argsmeta = []
    if kwargsmeta is None:
        kwargsmeta = {}
    if population_size is None:
        population_size = (len(argsmeta) + len(kwargsmeta)) * 10

    # Initizlize
    verbose = print_every is not None
    no_improvement_gens = 0
    iters_done = 0
    population = GAPopulation(argsmeta, kwargsmeta, population_size, True)

    try:
        with conditional_timeout(time, should_timeout=time is not None):

            # The optimization loop
            while True:
                # Begin a new generation

                if iters is not None and iters_done >= iters:
                    # We've reached the maximum number of iterations.
                    break

                if patience is not None and no_improvement_gens >= patience:
                    # No improvement for `patience` generations; stop the optimization.
                    break

                previous_best = population.best_fitness
                # Compute objective for each genome in population
                population.evaluate_fitness(fun)
                if population.best_fitness - previous_best > patience_tolerance:
                    # The population best has improved this generation by a sufficient
                    # amount.
                    no_improvement_gens = 0
                else:
                    # There was no improvement for this generation
                    no_improvement_gens += 1

                # Choose points from population for the mating pool
                parent_pairs = population.do_selection()

                # Create a new population from the mating pool
                population = population.crossover(parent_pairs)

                # Randomly mutate some genomes in the population
                population.mutate(mutation_prob)

                iters_done += 1
                if verbose and iters_done % print_every == 0:  # type: ignore
                    print(f"iter {iters_done} => fopt: {population.best_fitness:6.6f}")

                if callback is not None:
                    args_opt, kwargs_opt = population.best_genome.get_arg_vals()
                    end_prematurely = callback(
                        {
                            "args_opt": args_opt,
                            "kwargs_opt": kwargs_opt,
                            "nit": iters_done,
                            "fun": population.best_fitness,
                        }
                    )
                    if end_prematurely:
                        # The user has requested via callback to end the optimization.
                        break
    except EvaluationTimeoutError:
        if verbose:
            print("maximum time reached.")

    # Return "optimum" (best result found), and the arguments to `fun`
    # used to find it.
    args_opt, kwargs_opt = population.best_genome.get_arg_vals()
    if verbose:
        print("fopt:", population.best_fitness)
        print("optimal args:", args_opt, kwargs_opt)

    return population.best_fitness, args_opt, kwargs_opt
