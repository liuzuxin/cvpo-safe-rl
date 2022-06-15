from ray import tune

ENV_LIST = [
    'SafetyCarCircle-v0',
    # 'SafetyBallCircle-v0',
]

EXP_CONFIG = dict(
    env=tune.grid_search(ENV_LIST),
    timeout_steps=300,
    policy=tune.grid_search(["cvpo", "sac_lag", "ddpg_lag", "td3_lag"]),
    seed=tune.grid_search([0, 11, 22, 33, 44, 55, 66, 77, 88, 99]),
    cost_limit=tune.grid_search([20]),
    use_cost_decay=False,
    evaluate_episode_num=20,
    cost_start=300,
    cost_end=5,
    decay_epoch=200,
    warmup_steps=2000,
    batch_size=300,
    verbose=False,
    mode="train",
    device="cpu",
    threads=4,
    hidden_sizes=[256, 256],
    gamma=0.99,
    episode_rerun_num=28,
    mstep_iteration_num=29,
    sample_action_num=64,
    actor_lr=0.0064,
    critic_lr=0.0015,
    buffer_size=8000,
    # alpha_mean_scale=0,
    # alpha_var_scale=0,
)

SKIP_EXP_CONFIG = [{
    "env": 'SafetyCarCircle-v0',
    "cost_limit": 10
}, {
    "env": 'SafetyBallCircle-v0',
    "cost_limit": 20
}]

# EXP_NAME_KEYS = {"alpha_mean_scale":"alpha"}
EXP_NAME_KEYS = {}
DATA_DIR_KEYS = {"cost_limit": "cost"}
DATA_DIR_SUFFIX = '_benchmark'