import os
import time
import torch
import random
import pickle
import omnisafe
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from deep_sprl.util.parameter_parser import create_override_appendix
from omnisafe.models.actor import ActorBuilder

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class CurriculumType(Enum):
    GoalGAN = 1
    ALPGMM = 2
    SelfPaced = 3
    Default = 4
    Random = 5
    Wasserstein = 6
    ACL = 7
    PLR = 8
    VDS = 9

    def __str__(self):
        if self.goal_gan():
            return "goal_gan"
        elif self.alp_gmm():
            return "alp_gmm"
        elif self.self_paced():
            return "self_paced"
        elif self.wasserstein():
            return "wasserstein"
        elif self.default():
            return "default"
        elif self.acl():
            return "acl"
        elif self.plr():
            return "plr"
        elif self.vds():
            return "vds"
        else:
            return "random"

    def self_paced(self):
        return self.value == CurriculumType.SelfPaced.value

    def goal_gan(self):
        return self.value == CurriculumType.GoalGAN.value

    def alp_gmm(self):
        return self.value == CurriculumType.ALPGMM.value

    def default(self):
        return self.value == CurriculumType.Default.value

    def wasserstein(self):
        return self.value == CurriculumType.Wasserstein.value

    def random(self):
        return self.value == CurriculumType.Random.value

    def acl(self):
        return self.value == CurriculumType.ACL.value

    def plr(self):
        return self.value == CurriculumType.PLR.value

    def vds(self):
        return self.value == CurriculumType.VDS.value

    @staticmethod
    def from_string(string):
        if string == str(CurriculumType.GoalGAN):
            return CurriculumType.GoalGAN
        elif string == str(CurriculumType.ALPGMM):
            return CurriculumType.ALPGMM
        elif string == str(CurriculumType.SelfPaced):
            return CurriculumType.SelfPaced
        elif string == str(CurriculumType.Default):
            return CurriculumType.Default
        elif string == str(CurriculumType.Random):
            return CurriculumType.Random
        elif string == str(CurriculumType.Wasserstein):
            return CurriculumType.Wasserstein
        elif string == str(CurriculumType.ACL):
            return CurriculumType.ACL
        elif string == str(CurriculumType.PLR):
            return CurriculumType.PLR
        elif string == str(CurriculumType.VDS):
            return CurriculumType.VDS
        else:
            raise RuntimeError("Invalid string: '" + string + "'")

class AgentInterface(ABC):

    def __init__(self, learner, device):
        self.learner = learner
        self.device = device

    def estimate_value_r(self, observations):
        action, value_r, value_c, log_prob = self.learner._actor_critic.step(observations, deterministic=False)
        return value_r.detach().cpu().numpy()  

    def estimate_value_c(self, observations):
        action, value_r, value_c, log_prob = self.learner._actor_critic.step(observations, deterministic=False)
        return value_c.detach().cpu().numpy()

    def mean_policy_std(self):
        return self.learner._actor_critic.actor.std.detach().cpu().numpy()

    def get_action(self, observations):
        action, value_r, value_c, log_prob = self.learner._actor_critic.step(observations, deterministic=False)
        return action.detach().cpu().numpy()

    def save(self, log_dir):
        params = {
            'actor_critic': self.learner._actor_critic.state_dict() 
            if hasattr(self.learner._actor_critic, 'state_dict') 
            else self.learner._actor_critic,
        }
        torch.save(params, log_dir)

class Learner(Enum):
    PPO = 1
    SAC = 2
    PPOLag = 3

    def __str__(self):
        if self.ppo():
            return "PPO"
        elif self.sac():
            return "SAC"
        elif self.ppolag():
            return "PPOLag"
        else:
            return "Invalid"

    def ppo(self):
        return self.value == Learner.PPO.value

    def sac(self):
        return self.value == Learner.SAC.value
    
    def ppolag(self):
        return self.value == Learner.PPOLag.value
    
    def create_learner(self, env_id, custom_cfgs, wrapper_kwargs=None):
        model = omnisafe.Agent(str(self), env_id, custom_cfgs=custom_cfgs)
        if wrapper_kwargs is not None:
            model.agent._env._env.initialize_wrapper(**wrapper_kwargs)
        return model, AgentInterface(model, model.agent._device)

    def load_for_evaluation(self, model_path, obs_space, act_space, custom_cfgs, device='cpu'):
        actor_builder = ActorBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=custom_cfgs['model_cfgs']['actor']['hidden_sizes'],
            activation=custom_cfgs['model_cfgs']['actor']['activation'],
            weight_initialization_mode=custom_cfgs['model_cfgs']['weight_initialization_mode'],
        )
        actor = actor_builder.build_actor(custom_cfgs['model_cfgs']['actor_type'])
        model_params = torch.load(model_path, map_location=device)
        actor.load_state_dict(model_params['pi'])
        return actor

    @staticmethod
    def from_string(string):
        if string == str(Learner.PPO):
            return Learner.PPO
        elif string == str(Learner.SAC):
            return Learner.SAC
        elif string == str(Learner.PPOLag):
            return Learner.PPOLag
        else:
            raise RuntimeError("Invalid string: '" + string + "'")

class AbstractExperiment(ABC):
    APPENDIX_KEYS = {"default": ["DISCOUNT_FACTOR", "STEPS_PER_ITER", "LAM"],
                     CurriculumType.SelfPaced: ["DELTA", "KL_EPS", "DIST_TYPE", "INIT_VAR"],
                     CurriculumType.Wasserstein: ["DELTA", "METRIC_EPS"],
                     CurriculumType.GoalGAN: ["GG_NOISE_LEVEL", "GG_FIT_RATE", "GG_P_OLD"],
                     CurriculumType.ALPGMM: ["AG_P_RAND", "AG_FIT_RATE", "AG_MAX_SIZE"],
                     CurriculumType.Random: [],
                     CurriculumType.Default: [],
                     CurriculumType.ACL: ["ACL_EPS", "ACL_ETA"],
                     CurriculumType.PLR: ["PLR_REPLAY_RATE", "PLR_BETA", "PLR_RHO"],
                     CurriculumType.VDS: ["VDS_NQ", "VDS_LR", "VDS_EPOCHS", "VDS_BATCHES"]}

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed, device, view=False):
        self.device = device
        self.base_log_dir = base_log_dir
        self.parameters = parameters
        self.curriculum = CurriculumType.from_string(curriculum_name)
        self.learner = Learner.from_string(learner_name)
        self.seed = seed
        set_seed(seed)
        self.view = view
        self.process_parameters()

    @abstractmethod
    def create_experiment(self):
        pass

    @abstractmethod
    def get_env_name(self):
        pass

    @abstractmethod
    def create_self_paced_teacher(self):
        pass

    @abstractmethod
    def evaluate_learner(self, path):
        pass

    @abstractmethod
    def evaluate_training(self, path):
        pass

    def get_other_appendix(self):
        return ""

    @staticmethod
    def parse_max_size(val):
        if val == "None":
            return None
        else:
            return int(val)

    def process_parameters(self):
        allowed_overrides = {"DISCOUNT_FACTOR": float, "MAX_KL": float, "STEPS_PER_ITER": int,
                             "LAM": float, "AG_P_RAND": float, "AG_FIT_RATE": int,
                             "AG_MAX_SIZE": self.parse_max_size, "GG_NOISE_LEVEL": float, "GG_FIT_RATE": int,
                             "GG_P_OLD": float, "DELTA": float, "EPS": float, "MAZE_TYPE": str, "ACL_EPS": float,
                             "ACL_ETA": float, "PLR_REPLAY_RATE": float, "PLR_BETA": float, "PLR_RHO": float,
                             "VDS_NQ": int, "VDS_LR": float, "VDS_EPOCHS": int, "VDS_BATCHES": int,
                             "DIST_TYPE": str, "TARGET_TYPE": str, "KL_EPS": float,
                             "RALPH_IN": float, "RALPH": float, "RALPH_SCH": int, 
                             "EP_PER_UPDATE": int, "EP_PER_AUX_UPDATE": int, "INIT_VAR":float,
        }
        for key in sorted(self.parameters.keys()):
            if key not in allowed_overrides:
                raise RuntimeError("Parameter '" + str(key) + "'not allowed'")

            value = self.parameters[key]
            tmp = getattr(self, key)
            if isinstance(tmp, dict):
                tmp[self.learner] = allowed_overrides[key](value)
            else:
                setattr(self, key, allowed_overrides[key](value))

    def get_log_dir(self):
        override_appendix = create_override_appendix(self.APPENDIX_KEYS["default"], self.parameters)
        leaner_string = str(self.learner)
        key_list = self.APPENDIX_KEYS[self.curriculum]
        for key in sorted(key_list):
            tmp = getattr(self, key)
            if isinstance(tmp, dict):
                tmp = tmp[self.learner]
            leaner_string += "_" + key + "=" + str(tmp).replace(" ", "")

        return os.path.join(self.base_log_dir, self.get_env_name(), str(self.curriculum),
                            leaner_string + override_appendix + self.get_other_appendix(), "seed-" + str(self.seed))

    def train(self):
        model, omnisafe_log_dir = self.create_experiment()
        os.makedirs(self.get_log_dir(), exist_ok=True)
        with open(os.path.join(self.get_log_dir(), 'omnisafe_log_dir.txt'), 'w') as f:
            f.write(omnisafe_log_dir)
        model.learn()

    def evaluate(self, eval_type=0):
        log_dir = self.get_log_dir()
        iteration_dirs = [d for d in os.listdir(log_dir) if d.startswith("iteration-")]
        unsorted_iterations = np.array([int(d[len("iteration-"):]) for d in iteration_dirs])
        idxs = np.argsort(unsorted_iterations)
        sorted_iteration_dirs = np.array(iteration_dirs)[idxs].tolist()

        with open(os.path.join(log_dir, 'omnisafe_log_dir.txt'), 'r') as f:
            omnisafe_log_dir = f.read()
        omnisafe_saved_models = [d for d in os.listdir(os.path.join(omnisafe_log_dir, 'torch_save'))]
        unsorted_models = np.array([int(d[len("epoch-"):-3]) for d in omnisafe_saved_models])
        idxs = np.argsort(unsorted_models)
        sorted_models = np.array(omnisafe_saved_models)[idxs].tolist()
        
        # First evaluate the KL-Divergences if Self-Paced learning was used
        if self.curriculum.self_paced() and not os.path.exists(os.path.join(log_dir, "kl_divergences.pkl")):
            kl_divergences = []
            teacher = self.create_self_paced_teacher()
            for iteration_dir in sorted_iteration_dirs:
                iteration_log_dir = os.path.join(log_dir, iteration_dir)
                teacher.load(iteration_log_dir)
                kl_divergences.append(teacher.target_context_kl())

            kl_divergences = np.array(kl_divergences)
            with open(os.path.join(log_dir, "kl_divergences.pkl"), "wb") as f:
                pickle.dump(kl_divergences, f)

        performance_files = {
            0: "performance",
            1: "performance_hom",
        }
        for iteration_dir, saved_model in zip(sorted_iteration_dirs, sorted_models):
            print(f"Evaluating {iteration_dir} (eval_type={eval_type})")
            iteration_log_dir = os.path.join(log_dir, iteration_dir)
            performance_log_dir = os.path.join(iteration_log_dir, f"{performance_files[eval_type]}.npy")
            model_path = os.path.join(omnisafe_log_dir, 'torch_save', saved_model)
            eval_type_str = performance_files[eval_type][len("performance"):]
            # if not os.path.exists(performance_log_dir):
            if True:
                disc_rewards, eval_contexts, context_p, successful_eps, costs = self.evaluate_learner(
                    model_path=model_path,
                    eval_type=eval_type_str,
                )
                print(f"Evaluated {iteration_dir} (eval_type={eval_type}): {np.mean(disc_rewards)}")
                disc_rewards = np.array(disc_rewards)
                eval_contexts = np.array(eval_contexts)
                num_context = eval_contexts.shape[0]
                stats = np.ones((num_context, 1))*int(iteration_dir[len("iteration")+1:])
                stats = np.concatenate((stats, disc_rewards.reshape(-1, 1)), axis=1)
                stats = np.concatenate((stats, eval_contexts), axis=1)
                stats = np.concatenate((stats, context_p.reshape(-1, 1)), axis=1)
                stats = np.concatenate((stats, successful_eps.reshape(-1, 1)), axis=1)
                stats = np.concatenate((stats, costs.reshape(-1, 1)), axis=1)
                np.save(performance_log_dir, stats)

    def evaluate_training_performance(self, num_contexts=20):
        log_dir = self.get_log_dir()

        iteration_dirs = [d for d in os.listdir(log_dir) if d.startswith("iteration-")]
        unsorted_iterations = np.array([int(d[len("iteration-"):]) for d in iteration_dirs])
        idxs = np.argsort(unsorted_iterations)
        sorted_iteration_dirs = np.array(iteration_dirs)[idxs].tolist()

        with open(os.path.join(log_dir, 'omnisafe_log_dir.txt'), 'r') as f:
            omnisafe_log_dir = f.read()
        omnisafe_saved_models = [d for d in os.listdir(os.path.join(omnisafe_log_dir, 'torch_save'))]
        unsorted_models = np.array([int(d[len("epoch-"):]) for d in omnisafe_saved_models])
        idxs = np.argsort(unsorted_models)
        sorted_models = np.array(omnisafe_saved_models)[idxs].tolist()
        
        if self.curriculum.self_paced():
            for iteration_dir, saved_model in zip(sorted_iteration_dirs, sorted_models):
                print(f"Evaluating wrt context distribution in {iteration_dir}")
                iteration_log_dir = os.path.join(log_dir, iteration_dir)
                teacher = self.create_self_paced_teacher()
                teacher.load(iteration_log_dir)
                model_path = os.path.join(omnisafe_log_dir, 'torch_save', saved_model)
                training_contexts = np.array([teacher.sample() for _ in range(num_contexts)])
                performance_log_dir = os.path.join(iteration_log_dir, "performance_training.npy")
                if not os.path.exists(performance_log_dir):
                    disc_rewards, successful_eps, costs = self.evaluate_training(
                        model_path=model_path,
                        training_contexts=training_contexts,
                    )
                    context_p = teacher.context_dist.log_pdf_t(torch.from_numpy(training_contexts)).detach().numpy() 
                    print(f"Evaluated: {np.mean(disc_rewards)}")
                    disc_rewards = np.array(disc_rewards)
                    stats = np.ones((num_contexts, 1))*int(iteration_dir[len("iteration")+1:])
                    stats = np.concatenate((stats, disc_rewards.reshape(-1, 1)), axis=1)
                    stats = np.concatenate((stats, training_contexts), axis=1)
                    stats = np.concatenate((stats, context_p.reshape(-1, 1)), axis=1)
                    stats = np.concatenate((stats, successful_eps.reshape(-1, 1)), axis=1)
                    stats = np.concatenate((stats, costs.reshape(-1, 1)), axis=1)
                    np.save(performance_log_dir, stats)

        else:
            raise NotImplementedError("Evaluation of training performance is only implemented for Self-Paced learning")
