from environments.environment import BookingEnv
from trainer.dqn import train_dqn
from utils.params import Params

if __name__ == "__main__":
    env = BookingEnv()  # provide necessary parameters
    params = Params()

    train_dqn(env, params)
    train_dqn(env, params, dueling=True)
    train_dqn(env, params, double_dqn=False)
