import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
import logging
import argparse
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractionDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_df.iloc[idx]
        interaction_features = row['interaction_features']
        interaction_quality = row['interaction_quality']

        if self.transform:
            interaction_features = self.transform(interaction_features)

        return interaction_features, interaction_quality

class InteractionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(InteractionModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        predicted_quality = self.linear2(out)
        return predicted_quality

class AgentTrainer:
    def __init__(self, dataset_path, model_path, batch_size, learning_rate, num_epochs, input_dim, hidden_dim, output_dim):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InteractionModel(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def prepare_dataset(self):
        # Read and preprocess the dataset
        data_df = pd.read_csv(self.dataset_path)
        data_df = self._preprocess_data(data_df)

        # Split the dataset into training and validation sets
        train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42)

        return train_df, val_df

    def _preprocess_data(self, data_df):
        # Implement data preprocessing steps here
        # This is just a placeholder, you need to replace it with actual preprocessing logic
        return data_df

    def train(self):
        train_df, val_df = self.prepare_dataset()

        train_dataset = InteractionDataset(train_df)
        val_dataset = InteractionDataset(val_df)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0

            for batch_idx, (interaction_features, interaction_quality) in enumerate(train_loader):
                interaction_features = interaction_features.to(self.device)
                interaction_quality = interaction_quality.to(self.device)

                self.optimizer.zero_grad()
                predicted_quality = self.model(interaction_features)
                loss = self.criterion(predicted_quality, interaction_quality)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * interaction_features.size(0)

                if batch_idx % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}")

            train_loss /= len(train_dataset)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for interaction_features, interaction_quality in val_loader:
                    interaction_features = interaction_features.to(self.device)
                    interaction_quality = interaction_quality.to(self.device)

                    predicted_quality = self.model(interaction_features)
                    loss = self.criterion(predicted_quality, interaction_quality)
                    val_loss += loss.item() * interaction_features.size(0)

            val_loss /= len(val_dataset)

            logger.info(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save the model if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                logger.info(f"Validation loss improved, saving model to {self.model_path}")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        logger.info(f"Loaded model from {self.model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train the interaction quality prediction model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset CSV file')
    parser.add_ambigua('--model_path', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--input_dim', type=int, default=128, help='Input dimension for the model')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for the model')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension for the model')
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))

    trainer = AgentTrainer(args.dataset_path, args.model_path, args.batch_size, args.learning_rate, args.num_epochs,
                          args.input_dim, args.hidden_dim, args.output_dim)
    trainer.train()

if __name__ == '__main__':
    main()