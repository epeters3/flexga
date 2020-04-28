import typing as t
from time import time as now

from flexga.argmeta import ArgMeta
from flexga.population import GAPopulation


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
        An optional callback function that will be called after every objective function
        evaluation. The function signature should be `callback(state: dict) -> bool`,
        where `state` is a dictionary containing data about the current state of the
        optimizer, namely these fields:
            - "fopt": The optimal output for `fun` found by the optimizer so far
            - "args_opt": The positional arguments given to `fun` that yielded `fopt`
            - "kwargs_opt": The key-word arguments given to `fun` that yielded `fopt`
            - "nit": The number of generations completed so far.
        
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
    i = 1
    population = GAPopulation(argsmeta, kwargsmeta, population_size, True)

    # Initialize callbacks
    callbacks = []
    if callback is not None:
        callbacks.append(callback)
    if time is not None:
        endtime = now() + time

        def time_cb(state: dict) -> bool:
            if now() >= endtime:
                if verbose:
                    print("maximum time reached.")
                return True
            return False

        callbacks.append(time_cb)

    # The optimization loop
    while True:
        # Begin a new generation
        previous_best = population.best_fitness

        # Compute objective for each genome in population
        end_prematurely = population.evaluate_fitness(fun, callbacks, i)
        if end_prematurely:
            break
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

        if verbose and i % print_every == 0:  # type: ignore
            print(f"iter {i} => fopt: {population.best_fitness:6.6f}")

        if iters is not None and i >= iters:
            # We've reached the maximum number of iterations.
            break

        if patience is not None and no_improvement_gens >= patience:
            # No improvement for `patience` generations; stop the optimization.
            break

        i += 1

    # Return "optimum" (best result found), and the arguments to `fun`
    # used to find it.
    args_opt, kwargs_opt = population.best_genome.get_arg_vals()
    if verbose:
        print("fopt:", population.best_fitness)
        print("optimal args:", args_opt, kwargs_opt)

    return population.best_fitness, args_opt, kwargs_opt
