
# Deep Reinforcement Learning for Booking Simulation

This is deep reinforcement learning (DRL) project for booking simulation, where the DRL agent interacts with a custom environment, `BookingEnv`, to make optimal booking decisions by predicting prices from various providers.
The project utilizes the Tianshou library for reinforcement learning and PyTorch for neural network modeling.

## Table of Contents
- [Environment](#environment)
- [Installation](#installation)
- [Usage](#usage)
- [Structure](#structure)
- [Training](#training)
- [Results](#results)
- [License](#license)

## Environment


### `BookingEnv`: A Reinforcement Learning Framework for Price Predictions

The `BookingEnv` environment simulates the dynamics of price predictions from various providers over a temporal sequence, aiming to guide reinforcement learning agents towards optimal booking decisions.

#### Environment Components:

1. **Timeline**:
   - The environment operates over a temporal sequence $( T )$, where $( T )$ spans from `from_timestamp` to `to_timestamp`.
   - Predicted prices are updated at intervals $(\Delta t)$ defined by `step_seconds`.

2. **Prediction Model**:
   - The environment uses a predictive model, `prediction_model`, that forecasts the price matrix $( P )$ where each entry $( P_{i,j} )$ represents the predicted price from provider $( i )$ at time $( j )$ of the timeline.

3. **Agent's Decisions**:
   - At each time $t$, the agent observes the vector $P_{., t}$ (the predicted prices from all providers at time $t$ and decides):
     - To accept a price from a provider, resulting in action $a$ where $0 \leq a < \text{num_providers}$.
     - To wait for future offers, represented by action $a = \text{num_providers}$.

4. **Reward Function** $R(a, t)$:
   If the agent chooses a price from provider $a$ at time $t$:
     $R(a, t) = \text{large_reward}$ if $P_{a, t} = \min(P_{., t})$.
     Otherwise, $R(a, t) = \min(P_{., t}) - P_{a, t}$.
   If the agent decides to wait, the reward evolves with elapsed time 
   $$\tau:R(\text{num_providers}, \tau) = -\exp(0.01 \times \tau)$$.
   Exceeding the decision window results in $R(a, t) = \text{large_penalty}$.

#### Visualization:

- The `render` method graphically represents the matrix $P$, allowing users to visualize the temporal evolution of prices across providers.

#### Termination:

- An episode concludes, setting the `done` flag to `True`, either when the timeline reaches its terminal state or post a booking decision by the agent.





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

