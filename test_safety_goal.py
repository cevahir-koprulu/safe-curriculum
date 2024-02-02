
import safety_gymnasium
import deep_sprl
import deep_sprl.environments
import numpy as np
import time
from PIL import Image

contexts = [
    # np.array([-1.75, 2.5, 0.25]),
    # np.array([-1., 2.5, 0.25]),
    # np.array([1.75, 2.5, 0.25]),
    # np.array([1.75, 1.5, 0.25]),
    # np.array([1.75, -1., 0.25]),
    # np.array([1.75, -1.75, 0.25]),
    # np.array([0., -1.75, 0.25]),
    # np.array([1.5, -1.75, 0.25]),
    np.array([-1.25, -1.25, 0.25]),
    np.array([1.25, 1.25, 0.25]),
    # np.array([1.5, -2, 0.25]),
            ]
env_id = "SafetyPointContextualGoal1-v0" 
max_episode_steps = 1000

for context in contexts:
    print(f"\nContext: {context}")
    config = {'agent_name': "Car",
            'context': context}
    start = time.time()
    safety_gymnasium.__register_helper(
        env_id=env_id,
        entry_point='deep_sprl.environments.safety_goal.contextual_goal_level1:CustomBuilder',
        spec_kwargs={'config': config, 'task_id': env_id, 'render_mode': 'rgb_array', 'camera_name': 'fixedfar',
                     'width': 1000, 'height': 1000}, 
        max_episode_steps=max_episode_steps,
        
    )
    end = time.time()
    print("Time to register:", end - start)

    start = time.time()
    env = safety_gymnasium.make(env_id)
    end = time.time()
    print("Time to make:", end - start)

    start = time.time()
    s, i = env.reset()
    img = env.render()
    # Image.fromarray(img).save(f"videos/safety_goal/NEWSMALL_safety_goal0_context:{context}.png")

    print(env.action_space.sample())
    # print(env.observation_space)

    # # env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
    # for k in range(max_episode_steps):
    #     # s, a, d, t, i = env.step(env.action_space.sample())
    #     # img = env.render()
    #     # Image.fromarray(img).save(f"videos/safety_goal/safety_goal{k+1}.png")
    #     s, r, c, ter, tru, i = env.step(np.array([0.,0.5]))
    #     if env.task.goal_achieved:
    #         print(f"Step {k}")
    #         print(f"s:{s}")
    #         print(f"r:{r}")
    #         print(f"c:{c}")
    #         print(f"ter:{ter}")
    #         print(f"tru:{tru}")
    #         input(f"i:{i}")
    #     # if (k+1) % 100 == 0:
    #     #     img = env.render()
    #     #     Image.fromarray(img).save(f"videos/safety_goal/safety_goal{k+1}_context:{context}.png")
    #         # print(f"Step {k}")

    # end = time.time()
    # print("Time to simulate:", end - start)