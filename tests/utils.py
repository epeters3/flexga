def rosenbrock(x: float, y: float) -> float:
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def kw_rosenbrock(*, x: float, y: float) -> float:
    return rosenbrock(x, y)


def and_operator(x: bool, y: bool) -> float:
    """
    Poses the AND operator as an optimization problem.
    Useful only for testing that the genetic algorithm
    doesn't choke on a discrete function.
    """
    return 1.0 if x and y else 0.0
