# A Simple, Multi-Purpose Genetic Algorithm

[![Build Status](https://travis-ci.org/epeters3/flexga.svg?branch=master)](https://travis-ci.org/epeters3/flexga)

`flexga` is a flexible, multi-purpose elitist genetic algorithm useful for single objective optimization problems. It can simultaneously support float, integer, categorical, boolean, and float vector arguments. As such, it is a versatile tool for hyperparemter optimization in machine learning models, among other things.

## Getting Started

### Installation

```
pip install flexga
```

### Basic Usage

```python
from flexga import flexga
from flexga.utils import inverted
from flexga.argmeta import FloatArgMeta

def rosenbrock(x: float, y: float) -> float:
    """A classsic continuous optimization problem"""
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

# The goal of rosenbrock is to minimize. The genetic
# algorithm maximizes, so we invert the function's output.
objective = inverted(rosenbrock)

fopt, args_opt, _ = flexga(
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

## Annotating Objective Arguments

`flexga` can handle objective functions that take positional arguments (as seen above via the `argsmeta` parameter), as well as key-word arguments (via the `kwargsmeta` parameter, as seen below in the machine learning model hyperparameter optimization example). It can handle mixed-type arguments of several datatypes, which is one of its best features. The supported data types, alongside their dedicated annotation classes, are:

| Datatype                                                                    | Annotation Class                                          |
| --------------------------------------------------------------------------- | --------------------------------------------------------- |
| `float`                                                                     | `flexga.argmeta.FloatArgMeta(bounds, mutation_std)`       |
| `int`                                                                       | `flexga.argmeta.IntArgMeta(bounds, mutation_std)`         |
| `numpy.ndarray` vectors (must be 1D)                                        | `flexga.argmeta.FloatVectorArgMeta(bounds, mutation_std)` |
| `bool`                                                                      | `flexga.argmeta.BoolArgMeta()`                            |
| Categorical (one of a set of options, where the options can be of any type) | `flexga.argmeta.CategoricalArgMeta(options)`              |

See the constructor definitions for each of these annotation classes inside the `flexga.argmeta` module for details on what values they need.
