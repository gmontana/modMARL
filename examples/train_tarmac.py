"""Compatibility entry point; the complete recipe lives in modmarl.training.tarmac."""

from modmarl.training.tarmac import (
    _collect_episode as _collect_episode,
)
from modmarl.training.tarmac import (
    _evaluate as _evaluate,
)
from modmarl.training.tarmac import (
    _pad_episodes as _pad_episodes,
)
from modmarl.training.tarmac import (
    _team_reward as _team_reward,
)
from modmarl.training.tarmac import (
    main as main,
)
from modmarl.training.tarmac import (
    train as train,
)

if __name__ == "__main__":
    main()
