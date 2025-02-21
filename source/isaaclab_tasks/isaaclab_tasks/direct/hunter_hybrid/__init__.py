# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Hunter environment.

"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Register Gym environments.
##

gym.register(
    id="hunter_hybrid",
    entry_point=f"{__name__}.hunter_hybrid_env:HunterHybridEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hunter_hybrid_env:HunterHybridEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_hunter_cfg.yaml",
    },
)
