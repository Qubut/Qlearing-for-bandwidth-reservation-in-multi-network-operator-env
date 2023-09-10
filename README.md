
# Deep Reinforcement Learning for Booking Simulation

This is deep reinforcement learning (DRL) project for booking simulation, where the DRL agent interacts with a custom environment, `BookingEnv`, to make optimal booking decisions by predicting prices from various providers.
The project utilizes the Tianshou library for reinforcement learning and PyTorch for neural network modeling.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Structure](#structure)
- [Training](#training)
- [Results](#results)
- [License](#license)


## Installation

Ensure you have Python 3.11 installed. To set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/Qubut/Qlearing-for-bandwidth-reservation-in-multi-network-operator-env
   cd Qlearing-for-bandwidth-reservation-in-multi-network-operator-env
   ```

2. Install dependencies using [Poetry](https://python-poetry.org/docs/):
   ```bash
   poetry install
    ```
    or using `pip`

    Create a virtual environment (recommended) and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```
    then: 

    ```bash
    pip install -r requirements.txt
    ```


## Usage

To start training the DRL agent:

```shell
cd ./q_learing_for_bandwidth_reservation_in_multi_network_operator_env
poetry run python main.py

```
or if you have installed the pkgs without poetry 

```bash
cd ./q_learing_for_bandwidth_reservation_in_multi_network_operator_env
python main.py
```

This script trains three models:
- Double DQN
- Dueling DQN
- Single DQN

The results, including the models and rewards, are saved to the `./out` directory.

## Structure

- `environments/`: Contains the `BookingEnv` custom environment for the booking simulation.

- `models/`: Houses the neural network architectures, including the `DuelingNetwork`.

- `trainer/`: Contains the training logic and interactions with the Tianshou library.

- `utils/`: Utility functions and classes, including the `Params` data class for hyperparameters.

## Training

The training process involves using a Deep Q-Network (DQN) policy to train an agent in the BookingEnv environment. The training loop can be customized in the trainer/dqn.py module. Experiment with different hyperparameters and network architectures to achieve the best results.

## Results

Training results, including reward curves and model checkpoints, will be logged to the ./logs and ./out directories. Analyze these results to evaluate the performance of your trained agents.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

