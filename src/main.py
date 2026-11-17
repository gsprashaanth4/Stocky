import os
import random
import numpy as np

from copy import deepcopy

from req.data import download_stock
from req.features import calculate_features
from req.normalization import FeatureNormalizer
from req.Envir import StockEnvironment

from req.DQNNN import DQNNN
from req.Replay2 import Replay2


# ============================================================
# 1. CONFIGURATION
# ============================================================

TICKER = "AAPL"

INITIAL_CASH = 10000.0

LOOKBACK = 20
EPISODE_LENGTH = 252

TRAIN_END_DATE = "2022-01-01"



ACTIONS = np.array([
    0.0,
    0.25,
    0.50,
    0.75,
    1.0
])

STATE_DIM = LOOKBACK * 7 + 2
ACTION_DIM = len(ACTIONS)

action_counts = np.zeros(ACTION_DIM, dtype=int)

# Reproducibility
np.random.seed(42)
random.seed(42)


# ============================================================
# 2. DOWNLOAD HISTORICAL DATA
# ============================================================

print("\nDownloading data...")

raw_data = download_stock(
    ticker=TICKER,
    start="2000-01-01"
)

print(f"Downloaded {len(raw_data)} rows.")


# ============================================================
# 3. CALCULATE MARKET FEATURES
# ============================================================

print("\nCalculating features...")

feature_data = calculate_features(raw_data)

print(f"Feature rows: {len(feature_data)}")


# ============================================================
# 4. NORMALIZATION
# ============================================================

print("\nFitting feature normalizer...")

# Only use training-period data to fit normalization parameters.
train_features = feature_data[
    feature_data["Date"] < TRAIN_END_DATE
].copy()

normalizer = FeatureNormalizer()

normalizer.fit(train_features)

# Transform the feature dataset.
normalized_features = normalizer.transform(feature_data)

print("Normalization complete.")


# ============================================================
# 5. PREPARE TRAINING DATA
# ============================================================

print("\nPreparing training data...")

# Keep only dates belonging to the training period.
normalized_features = normalized_features[
    normalized_features["Date"] < TRAIN_END_DATE
].copy()

# Raw market data must contain exactly the same dates as
# the normalized feature data.
training_dates = normalized_features["Date"]

training_data = raw_data[
    raw_data["Date"].isin(training_dates)
].copy()

training_data = training_data.sort_values("Date").reset_index(drop=True)

normalized_features = normalized_features.sort_values(
    "Date"
).reset_index(drop=True)

# Verify alignment.
training_data["Date"] = np.asarray(
    training_data["Date"], dtype="datetime64[ns]"
)

normalized_features["Date"] = np.asarray(
    normalized_features["Date"], dtype="datetime64[ns]"
)

assert training_data["Date"].equals(
    normalized_features["Date"]
), "Raw data and normalized features are not aligned!"

print(f"Training rows: {len(training_data)}")

if len(training_data) <= LOOKBACK + EPISODE_LENGTH:
    raise ValueError("Not enough training data for the selected episode length.")


# ============================================================
# 6. INITIALIZE ENVIRONMENT
# ============================================================

print("\nInitializing environment...")

env = StockEnvironment(
    ticker=TICKER,
    data=training_data,
    normalized_features=normalized_features,
    lookback=LOOKBACK,
    episode_length=EPISODE_LENGTH,
    initial_cash=INITIAL_CASH
)

observation, info = env.reset()

assert observation.shape == (STATE_DIM,), (
    f"Expected observation shape {(STATE_DIM,)}, "
    f"received {observation.shape}"
)

print("Environment initialized.")
print("Initial information:", info)
print("Observation shape:", observation.shape)


# ============================================================
# 7. INITIALIZE DQN
# ============================================================

print("\nInitializing DQN...")

QValueNetwork = DQNNN([
    STATE_DIM,
    128,
    128,
    ACTION_DIM
])

targetNetwork = deepcopy(QValueNetwork)


# ============================================================
# 8. HYPERPARAMETERS
# ============================================================

episodes = 500

gamma = 0.99

epsilon = 1.0
epsilonMin = 0.05
epsilonDecay = 0.995

batchSize = 256
warmup = 3000

copyAt = 50

trainStep = 0
MSEloss = 0.0

replay = Replay2(batchSize)


# ============================================================
# 9. TRAINING
# ============================================================

print("\nStarting training...\n")
allocations = []

for episode in range(episodes):

    observation, info = env.reset()

    terminated = False
    truncated = False

    episodeReward = 0.0
    MSEloss = 0.0

    while not (terminated or truncated):

        # ----------------------------------------------------
        # ACTION SELECTION
        # ----------------------------------------------------

        qValues = QValueNetwork.interact(observation)

        if np.random.random() < epsilon:

            action_idx = np.random.randint(ACTION_DIM)

        else:

            action_idx = int(np.argmax(qValues[0]))

        # Convert action index to target allocation.
        allocation = ACTIONS[action_idx]
        action_counts[action_idx] += 1


        # ----------------------------------------------------
        # ENVIRONMENT STEP
        # ----------------------------------------------------

        prevObs = observation

        observation, reward, terminated, truncated, info = \
            env.step(allocation)

        allocations.append(info["allocation"])

        
        # ----------------------------------------------------
        # EXPERIENCE REPLAY
        # ----------------------------------------------------

        done = terminated or truncated

        replay.jotDown(
            prevObs,
            action_idx,
            reward,
            observation,
            done
        )

        episodeReward += reward


        # ----------------------------------------------------
        # DQN TRAINING
        # ----------------------------------------------------

        if len(replay.samples) >= warmup:

            states, actions, rewards, next_states, dones = \
                replay.prepareData()

            actions = actions.astype(np.int64)


            # Current network predictions
            Qvals, _ = QValueNetwork.getQvals(states)


            # Target network predictions
            nQvals, _ = targetNetwork.getQvals(next_states)


            # Maximum future Q-value
            maxNextQ = np.max(nQvals, axis=1)


            # Bellman equation
            targets = (
                rewards
                + gamma * (1.0 - dones) * maxNextQ
            )


            # Extract Q-values for selected actions
            indices = np.arange(len(actions))

            predQvals = Qvals[indices, actions]


            # Mean squared error
            sqError = (predQvals - targets) ** 2

            MSEloss = np.mean(sqError)


            # Gradient of MSE
            dLdP = (
                2.0 / len(actions)
            ) * (predQvals - targets)


            # Gradient with respect to network outputs
            dPdZ = np.zeros_like(Qvals)

            dPdZ[indices, actions] = dLdP


            # Backpropagation
            QValueNetwork.backward(dPdZ)

            QValueNetwork.optimize()


            # Update training counter
            trainStep += 1


            # Update target network periodically
            if trainStep % copyAt == 0:

                targetNetwork = deepcopy(QValueNetwork)


    # ========================================================
    # 10. EPSILON DECAY
    # ========================================================

    epsilon = max(
        epsilonMin,
        epsilon * epsilonDecay
    )


    # ========================================================
    # 11. LOGGING
    # ========================================================
    avg_allocation = np.mean(allocations)
    print(
        f"Episode: {episode + 1:04d} | "
        f"Reward: {episodeReward:.5f} | "
        f"Equity: {info['equity_after']:.2f} | "
        f"Loss: {MSEloss:.6f} | "
        # f"Q-max: {np.max(Qvals):.4f} | "
        # f"Q-min: {np.min(Qvals):.4f} | "
        f"Epsilon: {epsilon:.4f} | "
        f"Avg allocation: {avg_allocation:.4f}"
    )

    


# ============================================================
# 12. SAVE NETWORK
# ============================================================

print("\nTraining complete.")

os.makedirs("models", exist_ok=True)

QValueNetwork.save(
    "models/stocky_aapl_dqn.npz"
)

print("Model saved to models/stocky_aapl_dqn.npz")


# ============================================================
# 13. CLOSE ENVIRONMENT
# ============================================================

if hasattr(env, "close"):
    env.close()