from gym.envs.registration import register

register(
    id='ContextualPointMass2D-v1',
    max_episode_steps=200,
    entry_point='deep_sprl.environments.point_mass.contextual_point_mass_2d:ContextualPointMass2D'
)

register(
    id='ContextualSafetyPointMass2D-v1',
    max_episode_steps=200,
    entry_point='deep_sprl.environments.safety_point_mass.contextual_safety_point_mass_2d:ContextualSafetyPointMass2D'
)

register(
    id='ContextualLunarLander-v1',
    max_episode_steps=1000,
    reward_threshold=200,
    entry_point='deep_sprl.environments.lunar_lander.contextual_lunar_lander:ContextualLunarLander'
)

register(
    id='ContextualLunarLander2D-v1',
    max_episode_steps=1000,
    reward_threshold=200,
    entry_point='deep_sprl.environments.lunar_lander.contextual_lunar_lander_2d:ContextualLunarLander2D'
)