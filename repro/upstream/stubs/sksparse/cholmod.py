"""Import-only stub of ``sksparse.cholmod`` (see package docstring)."""


class _UnavailableError(RuntimeError):
    pass


class Factor:
    """Placeholder for scikit-sparse's CHOLMOD factor object."""

    def __init__(self, *args, **kwargs):
        raise _UnavailableError(
            "sksparse/skparse is a stub: CHOLMOD (SuiteSparse) is not installed. "
            "Theseus dense solvers are available; the CHOLMOD sparse solver is not."
        )


def analyze(*args, **kwargs):  # pragma: no cover
    raise _UnavailableError("CHOLMOD is not installed (sksparse stub).")


def cholesky(*args, **kwargs):  # pragma: no cover
    raise _UnavailableError("CHOLMOD is not installed (sksparse stub).")


def analyze_AAt(*args, **kwargs):  # pragma: no cover
    raise _UnavailableError("CHOLMOD is not installed (sksparse stub).")
