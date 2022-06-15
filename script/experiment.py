import os.path as osp

import ray
from ray import tune

from safe_rl.util.run_util import load_config
from safe_rl.runner import Runner

CONFIG_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "config")


def gen_exp_name(config: dict):
    name = config["policy"]
    for k in EXP_NAME_KEYS:
        name += '_' + EXP_NAME_KEYS[k] + '_' + str(config[k])
    return name


def gen_data_dir_name(config: dict):
    name = config["env"]
    for k in DATA_DIR_KEYS:
        name += '_' + DATA_DIR_KEYS[k] + '_' + str(config[k])
    return name + DATA_DIR_SUFFIX


def trial_name_creator(trial):
    config = trial.config
    name = config["env"]
    for k in DATA_DIR_KEYS:
        name += '_' + DATA_DIR_KEYS[k] + '_' + str(config[k])
    return name + DATA_DIR_SUFFIX + '_' + config["policy"]


def skip_exp(config):
    '''
    determine if we should skip this exp
    '''
    for skip in SKIP_EXP_CONFIG:
        state = True if len(skip) > 0 else False
        for k in skip:
            state = state and (skip[k] == config[k])
        if state:
            return True
    return False


def trainable(config):
    if skip_exp(config):
        '''
        Skip this exp if it satisfies some criterion
        '''
        return False

    config["exp_name"] = gen_exp_name(config)
    config["data_dir"] = gen_data_dir_name(config)
    policy = config["policy"]

    if policy == "cvpo":
        config_path = osp.join(CONFIG_DIR, "config_cvpo.yaml")
    else:
        config_path = osp.join(CONFIG_DIR, "config_baseline.yaml")
    default_config = load_config(config_path)

    # replace the default config with search configs
    for k, v in config.items():
        if k in default_config:
            default_config[k] = v
        if k in default_config[policy].keys():
            default_config[policy][k] = v
        if k in default_config[policy]["worker_config"].keys():
            default_config[policy]["worker_config"][k] = v

    runner = Runner(**default_config)
    runner.train()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('policy', nargs='*')
    parser.add_argument('--env',
                        '-e',
                        type=str,
                        help='experiment env name, button, circle, goal, or push')
    parser.add_argument('--cpus',
                        '--cpu',
                        type=int,
                        default=4,
                        help='maximum cpu resources for ray')
    parser.add_argument('--threads',
                        '--thread',
                        type=int,
                        default=4,
                        help='maximum threads resources per trial')

    args = parser.parse_args()

    ray.init(num_cpus=args.cpus)

    env = args.env.lower()
    if env == 'button':
        from button import ENV_LIST, EXP_CONFIG, SKIP_EXP_CONFIG, EXP_NAME_KEYS, DATA_DIR_KEYS, DATA_DIR_SUFFIX
    elif env == 'goal':
        from goal import *
    elif env == 'circle':
        from circle import *

    EXP_CONFIG["threads"] = args.threads
    EXP_CONFIG["policy"] = tune.grid_search(args.policy)

    experiment_spec = tune.Experiment(
        args.env,
        trainable,
        config=EXP_CONFIG,
        resources_per_trial={
            "cpu": args.threads,
            "gpu": 0
        },
        trial_name_creator=trial_name_creator,
    )

    tune.run_experiments(experiment_spec)
