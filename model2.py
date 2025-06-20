# === Imports ===
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# === Predictive Model for Free Energy ===
class PredictiveModel(nn.Module):
    def __init__(self, state_dim, image_dim, output_dim):
        super(PredictiveModel, self).__init__()
        self.state_fc1 = nn.Linear(state_dim, 128)
        self.state_fc2 = nn.Linear(128, 64)

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc_image = nn.Linear(128 * 8 * 8, 256)

        self.fc_combined = nn.Linear(64 + 256, 128)
        self.fc_out = nn.Linear(128, output_dim)

    def forward(self, state, image):
        state_x = torch.relu(self.state_fc1(state))
        state_x = torch.relu(self.state_fc2(state_x))

        image_x = torch.relu(self.conv1(image))
        image_x = torch.relu(self.conv2(image_x))
        image_x = torch.relu(self.conv3(image_x))
        image_x = image_x.view(image_x.size(0), -1)
        image_x = torch.relu(self.fc_image(image_x))

        combined = torch.cat((state_x, image_x), dim=1)
        combined = torch.relu(self.fc_combined(combined))
        predicted_state = self.fc_out(combined)
        return predicted_state


# === Active Inference Agent ===
class ActiveInferenceAgent:
    def __init__(self, state_dim, image_dim, output_dim):
        self.model = PredictiveModel(state_dim, image_dim, output_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def predict_next_state(self, state, image):
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            image_tensor = image_tensor.permute(0, 3, 1, 2)
            predicted_state = self.model(state_tensor, image_tensor)
        return predicted_state.numpy().flatten()

    def compute_prediction_error(self, actual_state, predicted_state):
        return np.linalg.norm(actual_state - predicted_state)

    def update_model(self, state, image, next_state):
        self.model.train()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        image_tensor = image_tensor.permute(0, 3, 1, 2)

        self.optimizer.zero_grad()
        predicted_state = self.model(state_tensor, image_tensor)
        loss = self.criterion(predicted_state, next_state_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()


# === Beispiel für Integration in das Environment (z. B. in step() oder RL Loop) ===
def compute_intrinsic_reward(agent, current_state, image, next_state, beta=0.01):
    predicted = agent.predict_next_state(current_state, image)
    free_energy = agent.compute_prediction_error(next_state, predicted)
    return beta * free_energy


# === Initialisierung (einmalig zu Beginn deines Notebooks) ===
state_dim = 5  # z.B. [x, y, yaw, goal_x, goal_y]
image_dim = (4, 64, 64)
output_dim = 3  # z.B. [x_next, y_next, yaw_next]
agent = ActiveInferenceAgent(state_dim, image_dim, output_dim)

# === Beispiel für Anwendung ===
# current_state = obs["state"]
# camera_image = obs["camera"]
# next_state = next_obs["state"][:3]
# intrinsic_reward = compute_intrinsic_reward(agent, current_state, camera_image, next_state)
# total_reward = extrinsic_reward + intrinsic_reward
