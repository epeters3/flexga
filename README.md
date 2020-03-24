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

## Example: Machine Learning Model Hyperparameter Optimization

Here is an example of using `flexga` to perform hyperparameter optimization of both continuous and integer hyperparameters for a decision tree classifier. `flexga` optimizes the F1 Macro metric as computed over a validation set on the digits problem.

```python
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from flexga import flexga
from flexga.argmeta import IntArgMeta, FloatArgMeta

# Make the train/validation split of the data.
dataset = load_digits()
X_train, X_val, y_train, y_val = train_test_split(
    dataset["data"], dataset["target"], test_size=0.33
)

# We use this model throughout the optimization process.
model = DecisionTreeClassifier()

# Get a baseline to compare the optimized result to.
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print(
    "baseline for default hyperparameters:",
    f1_score(y_val, y_pred, average="macro")
)

# This function accepts hyperparameter values, trains
# the model on the training set with those values,
# and returns F1 Macro computed over the validation
# set for the trained model. It is of the form
# `objective_function(design_variables) -> objective`,
# which is what the optimizer wants.
def objective(*args, **kwargs) -> float:
    # Set the hyperparameters - the things we're optimizing.
    model.set_params(**kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average="macro")


# Optimize the decision tree's hyperparameters over the
# validation set. `fopt`, the optimal F1 Macro found,
# should be higher than the baseline.
fopt, _, kwargs_opt = flexga(
    objective,
    kwargsmeta={
        # These are the things the genetic algorithm will optimize;
        # the decision tree's hyperparameters.
        "min_samples_split": IntArgMeta((2, 75), 3),
        "min_samples_leaf": IntArgMeta((1, 75), 3),
        "min_weight_fraction_leaf": FloatArgMeta((0.0, 0.5), 0.025),
        "min_impurity_decrease": FloatArgMeta((0.0, 1.0), 0.05),
        "ccp_alpha": FloatArgMeta((0.0, 1.0), 0.05),
    },
    # Will stop when the optimizer has not improved upon the best
    # fitness for 20 generations.
    patience=20,
    verbose=True,
)
```
