from unittest import TestCase
from time import time

from flexga import flexga
from flexga.utils import inverted
from flexga.argmeta import FloatArgMeta, BoolArgMeta
from tests.utils import rosenbrock, kw_rosenbrock, and_operator


class TestFlexGA(TestCase):
    def test_can_optimize_with_args(self) -> None:
        fopt, _, _ = flexga(
            inverted(rosenbrock),
            argsmeta=[FloatArgMeta((-50, 50), 1.0), FloatArgMeta((-50, 50), 1.0)],
            iters=1000,
            patience=None,
        )
        # With 1000 iterations, we should easily be within
        # .01 of the true optimum.
        assert fopt >= -0.01

    def test_can_optimize_with_kwargs(self) -> None:
        fopt, _, _ = flexga(
            inverted(kw_rosenbrock),
            kwargsmeta={
                "x": FloatArgMeta((-50, 50), 1.0),
                "y": FloatArgMeta((-50, 50), 1.0),
            },
            iters=1000,
            patience=None,
        )
        # With 1000 iterations, we should easily be within
        # .01 of the true optimum.
        assert fopt >= -0.01

    def test_doesnt_error_on_discrete(self) -> None:
        fopt, _, _ = flexga(
            and_operator, argsmeta=[BoolArgMeta(), BoolArgMeta()], patience=10,
        )
        assert fopt == 1

    def test_can_use_callback(self) -> None:
        fopt, _, _ = flexga(
            inverted(rosenbrock),
            argsmeta=[FloatArgMeta((-50, 50), 1.0), FloatArgMeta((-50, 50), 1.0)],
            iters=1000,
            patience=None,
            # End prematurely after 10 iterations.
            callback=lambda state: True if state["nit"] >= 10 else False,
        )

    def test_timeout_successfully(self) -> None:
        self.total_iters = 0

        def get_iters(state) -> bool:
            self.total_iters = state["nit"]
            return False

        start = time()
        fopt, _, _ = flexga(
            inverted(rosenbrock),
            argsmeta=[FloatArgMeta((-50, 50), 1.0), FloatArgMeta((-50, 50), 1.0)],
            # Should take much longer than 1 second
            iters=int(1e10),
            patience=None,
            # End prematurely after 1 second
            time=1,
            callback=get_iters,
        )
        end = time() - start
        # Timeout was 1 so should be long done at 2
        assert end < 2
        assert self.total_iters > 0 and self.total_iters < 1e10
        assert fopt > -float("inf")
