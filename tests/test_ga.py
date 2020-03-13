from unittest import TestCase

from flexga import ga
from flexga.utils import inverted
from flexga.argmeta import FloatArgMeta, BoolArgMeta
from tests.utils import rosenbrock, kw_rosenbrock, and_operator


class TestGA(TestCase):
    def test_can_optimize_with_args(self) -> None:
        """
        `ga` should be able to run on the basic two
        argument rosenbrock problem.
        """
        fopt, _, _ = ga(
            inverted(rosenbrock),
            argsmeta=[FloatArgMeta((-50, 50), 1.0), FloatArgMeta((-50, 50), 1.0)],
            iters=1000,
        )
        # With 1000 iterations, we should easily be within
        # .01 of the true optimum.
        assert fopt >= -0.01

    def test_can_optimize_with_kwargs(self) -> None:
        fopt, _, _ = ga(
            inverted(kw_rosenbrock),
            kwargsmeta={
                "x": FloatArgMeta((-50, 50), 1.0),
                "y": FloatArgMeta((-50, 50), 1.0),
            },
            iters=1000,
        )
        # With 1000 iterations, we should easily be within
        # .01 of the true optimum.
        assert fopt >= -0.01

    def test_doesnt_error_on_discrete(self) -> None:
        fopt, _, _ = ga(
            and_operator, argsmeta=[BoolArgMeta(), BoolArgMeta()], iters=10,
        )
        assert fopt == 1
