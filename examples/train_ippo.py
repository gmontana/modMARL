"""Compatibility entry point; the complete recipe lives in modmarl.training.ippo."""

from modmarl.training.ippo import (
    _evaluate as _evaluate,
)
from modmarl.training.ippo import (
    main as main,
)
from modmarl.training.ippo import (
    train as train,
)

if __name__ == "__main__":
    main()
