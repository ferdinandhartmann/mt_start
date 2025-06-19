import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PredictiveModel(nn.Module):
    def __init__(self, state_dim, image_dim, output_dim):
        super(PredictiveModel, self).__init__()

        # Define layers for state input (fully connected)
        self.state_fc1 = nn.Linear(state_dim, 128)
        self.state_fc2 = nn.Linear(128, 64)

        # Define layers for camera image input (convolutional layers)
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=32, kernel_size=3, stride=2, padding=1
        )  # 64x64x4 -> 32x32x32
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )  # 32x32x32 -> 16x16x64
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=2, padding=1
        )  # 16x16x64 -> 8x8x128
        self.fc_image = nn.Linear(128 * 8 * 8, 256)  # Flatten the image feature map

        # Combine the features from both state and image, and predict next state
        self.fc_combined = nn.Linear(64 + 256, 128)
        self.fc_out = nn.Linear(
            128, output_dim
        )  # Final output for predicted next state

    def forward(self, state, image):
        # Process state input
        state_x = torch.relu(self.state_fc1(state))
        state_x = torch.relu(self.state_fc2(state_x))

        # Process camera image input (assuming the image is already in shape [batch_size, 4, 64, 64])
        image_x = torch.relu(self.conv1(image))
        image_x = torch.relu(self.conv2(image_x))
        image_x = torch.relu(self.conv3(image_x))
        image_x = image_x.reshape(image_x.size(0), -1)  # Flatten the feature map
        image_x = torch.relu(self.fc_image(image_x))

        # Combine state and image features
        combined = torch.cat((state_x, image_x), dim=1)

        # Final prediction of next state
        combined = torch.relu(self.fc_combined(combined))
        predicted_state = self.fc_out(combined)

        return predicted_state


class ActiveInferenceAgent:
    def __init__(self, state_dim, image_dim, output_dim):
        # Initialize the model, loss function, and optimizer
        self.model = PredictiveModel(state_dim, image_dim, output_dim)
        self.criterion = (
            nn.MSELoss()
        )  # Mean Squared Error (for minimizing prediction error)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def predict_next_state(self, state, image):
        """
        Predict the next state given the current state and camera image.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        # Transpose the image from [batch_size, height, width, channels] to [batch_size, channels, height, width]
        image_tensor = image_tensor.permute(
            0, 3, 1, 2
        )  # Change shape from [1, 64, 64, 4] to [1, 4, 64, 64]

        predicted_state = self.model(state_tensor, image_tensor)
        return predicted_state.detach().numpy().flatten()

    def compute_prediction_error(self, actual_state, predicted_state):
        """
        Compute the prediction error (free energy) by comparing actual and predicted states.
        """
        error = np.linalg.norm(actual_state - predicted_state)
        return error

    def update_model(self, state, image, next_state):
        """
        Update the model using the current state, camera image, and next state.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # Transpose the image from [batch_size, height, width, channels] to [batch_size, channels, height, width]
        image_tensor = image_tensor.permute(
            0, 3, 1, 2
        )  # Change shape from [1, 64, 64, 4] to [1, 4, 64, 64]

        # Zero the gradients before backward pass
        self.optimizer.zero_grad()

        # Forward pass
        predicted_state = self.model(state_tensor, image_tensor)

        # Compute the loss (MSE loss)
        loss = self.criterion(predicted_state, next_state_tensor)

        # Backward pass and optimization step
        loss.backward()
        self.optimizer.step()

        return loss.item()


# Example usage:
state_dim = 5  # [x, y, yaw, x_goal, y_goal]
image_dim = (4, 64, 64)  # 4 channels (RGBA), 64x64 pixels
output_dim = 3  # Predicted state is [x_next, y_next, yaw_next]
agent = ActiveInferenceAgent(state_dim, image_dim, output_dim)

# Simulate a step in the environment:
current_state = np.array([1.0, 2.0, 0.5, 3.0, 4.0])  # Example current state
camera_image = np.random.rand(64, 64, 4)  # Example random camera image (RGBA)
next_state = np.array([1.2, 2.1, 0.55])  # Example next state

# Predict the next state based on the current state and camera image
predicted_state = agent.predict_next_state(current_state, camera_image)

# Compute the prediction error (free energy)
prediction_error = agent.compute_prediction_error(next_state, predicted_state)
print(f"Prediction Error (Free Energy): {prediction_error}")

# Update the model (train the model on this data)
loss = agent.update_model(current_state, camera_image, next_state)
print(f"Model update loss: {loss}")
