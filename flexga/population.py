import typing as t

from flexga.argmeta import ArgMeta
from flexga.utils import grouper, shuffle
from flexga.genome import Genome


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

    def evaluate_fitness(
        self,
        objective: t.Callable,
        callbacks: t.Optional[t.Iterable[t.Callable]],
        gen_num: int,
    ) -> bool:
        for genome in self.members:

            genome.evaluate_fitness(objective)
            if genome.fitness > self.best_fitness:
                # We've found a new best
                self.best_fitness = genome.fitness
                self.best_genome = genome

            if callbacks is not None:
                args_opt, kwargs_opt = self.best_genome.get_arg_vals()
                end_prematurely = any(
                    cb(
                        {
                            "args_opt": args_opt,
                            "kwargs_opt": kwargs_opt,
                            "nit": gen_num,
                            "fun": self.best_fitness,
                        }
                    )
                    for cb in callbacks
                )
                if end_prematurely:
                    # User has requested to exit the optimization early.
                    return True

        return False

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
