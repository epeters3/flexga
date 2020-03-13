# A Simple, Multi-Purpose Genetic Algorithm

[![Build Status](https://travis-ci.org/epeters3/flexga.svg?branch=master)](https://travis-ci.org/epeters3/flexga)

`flexga` is a flexible, multi-purpose elitist genetic algorithm that can simultaneously support float, integer, categorical, boolean, and float vector arguments.

## Getting Started

```
git clone https://github.com/epeters3/flexga.git
cd flexga
```

```python
from flexga import ga
from flexga.utils import inverted
from flexga.argmeta import FloatArgMeta

def rosenbrock(x: float, y: float) -> float:
    """A classsic continuous optimization problem"""
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

# The goal of rosenbrock is to minimize. The genetic
# algorithm maximizes, so we invert the function's output.
objective = inverted(rosenbrock)

fopt, args_opt, _ = ga(
    objective,
    # We must specify annotations for rosenbrock's arguments,
    # in this case so the optimizer knows what the bounds are
    # for each input, and so it knows what distribution to
    # sample mutation values from.
    argsmeta = [
        FloatArgMeta(bounds=(-50, 50), mutation_std=1.0),
        FloatArgMeta(bounds=(-50, 50), mutation_std=1.0)
    ]
    iters = 500,
)

# The best value the optimizer found for the objective.
print(fopt) # 0.0

# The arguments that give the objective its optimal value
# i.e. `rosenbrock(*args_opt) == fopt`.
print(args_opt) # [1.0, 1.0]
```

`ga` can handle positional arguments (as seen above via the `argsmeta` parameter), as well as key-word arguments (via the `kwargsmeta` parameter). It can handle mixed-type arguments of several datatypes, which is one of its best features. The supported data types, alongside their dedicated annotation class, are:

| Datatype                                                                    | Annotation Class                |
| --------------------------------------------------------------------------- | ------------------------------- |
| `float`                                                                     | `ga.argmeta.FloatArgMeta`       |
| `int`                                                                       | `ga.argmeta.IntArgMeta`         |
| `numpy.ndarray` vectors (must be 1D)                                        | `ga.argmeta.FloatVectorArgMeta` |
| `bool`                                                                      | `ga.argmeta.BoolArgMeta`        |
| Categorical (one of a set of options, where the options can be of any type) | `ga.argmeta.CategoricalArgMeta` |

See the constructor definitions for each of these annotation classes inside the `ga.argmeta` module for details on what values they need.
