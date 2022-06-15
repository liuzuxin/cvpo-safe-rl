from ray import tune

ENV_LIST = [
    'Safexp-CarButton2-v0',
    'Safexp-PointButton2-v0',
]

EXP_CONFIG = dict(
    env=tune.grid_search(ENV_LIST),
    timeout_steps=400,
    policy=tune.grid_search(["cvpo", "sac_lag", "ddpg_lag", "td3_lag"]),
    seed=tune.grid_search([0, 11, 22, 33, 44, 55, 66, 77, 88, 99]),
    #seed=tune.grid_search([0, 11, 22, 33, 44]),
    cost_limit=tune.grid_search([10]),
    use_cost_decay=False,
    evaluate_episode_num=20,
    cost_start=300,
    cost_end=5,
    decay_epoch=200,
    warmup_steps=4000,
    batch_size=300,
    verbose=False,
    mode="train",
    device="cpu",
    threads=4,
    hidden_sizes=[256, 256],
    gamma=0.99,
)

SKIP_EXP_CONFIG = []

EXP_NAME_KEYS = {}
DATA_DIR_KEYS = {"cost_limit": "cost"}
DATA_DIR_SUFFIX = '_benchmark'