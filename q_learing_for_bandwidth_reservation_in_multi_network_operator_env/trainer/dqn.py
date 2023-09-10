import os
import torch
import numpy as np
from tianshou.policy import DQNPolicy
from tianshou.data import Collector, ReplayBuffer
from tianshou.trainer import offpolicy_trainer
from torch.utils.tensorboard import SummaryWriter
from models.networks import build_network_for
from utils.params import Params


def train_dqn(env, dueling=False, double_dqn=True):
    params = Params()
    net = build_network_for(env, dueling=dueling)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    policy = DQNPolicy(
        model=net,
        optim=optimizer,
        target_update_freq=params.target_update_freq,  # Adjust frequency as needed
        is_double=double_dqn,
    )

    buffer = ReplayBuffer(size=params.replay_buffer_size)
    train_collector = Collector(policy, env, buffer)
    test_collector = Collector(policy, env)

    algo_name = "DQN"
    if dueling:
        algo_name = "Dueling" + algo_name
    if double_dqn:
        algo_name = "Double" + algo_name

    log_path = f"./logs/{env.spec.id}_{algo_name}"
    writer = SummaryWriter(log_dir=log_path)

    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=10,
        step_per_epoch=1000,
        collect_per_step=10,
        test_num=100,
        batch_size=64,
        train_fn=policy.sync_weight,
        writer=writer,
    )

    if not os.path.exists("./out"):
        os.makedirs("./out")
    np.save(
        f"./out/{env.spec.id}_{algo_name}_train_rewards.npy", result["train_rewards"]
    )
    np.save(f"./out/{env.spec.id}_{algo_name}_test_rewards.npy", result["test_rewards"])

    model_save_dir = f"./out/models/{env.spec.id}_{algo_name}"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = f"{model_save_dir}/model.pth"
    torch.save(policy.state_dict(), model_save_path)

    return result
