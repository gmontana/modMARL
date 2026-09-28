"""Scenario package for the vendored particle environment.

modMARL: the original ``imp``-based ``load()`` helper was removed (``imp`` no longer exists in
Python 3.12); ``marl_envs.particle._load_scenario_module`` loads scenarios by file path.
"""
