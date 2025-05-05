"""
Time series model for IPL match prediction using LSTM with improved features.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class IPLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class IPLLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super(IPLLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc1 = nn.Linear(hidden_size, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take only the last output

        # Fully connected layers
        x = self.fc1(lstm_out)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.sigmoid(x)

        return x


class IPLTimeSeriesModel:
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.scaler = StandardScaler()
        self.sequence_length = 8  # Reduced sequence length
        self.team_form_length = 5  # Increased form length
        self.feature_scaler = MinMaxScaler()  # Additional scaler for features
        self.imputer = SimpleImputer(
            strategy="mean"
        )  # Imputer for handling missing values
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        """Load and preprocess the data."""
        try:
            # Load the data
            df = pd.read_csv("data/processed/ipl_matches_processed.csv")

            # Add form features
            df = self.add_team_form_features(df)

            # Create sequences
            feature_cols = [
                col
                for col in df.columns
                if col.endswith("_encoded")
                or col in ["year", "month", "day"]
                or col.endswith("_form")
                or col.endswith("_wins")
                or col.endswith("_score")
            ]

            X = df[feature_cols].values
            y = df["winner"].values

            # Calculate the number of sequences
            n_samples = len(X)
            n_features = len(feature_cols)
            n_sequences = n_samples - self.sequence_length + 1

            # Create sequences
            X_sequences = np.zeros((n_sequences, self.sequence_length, n_features))
            y_sequences = np.zeros(n_sequences)

            for i in range(n_sequences):
                X_sequences[i] = X[i : i + self.sequence_length]
                y_sequences[i] = y[i + self.sequence_length - 1]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_sequences, y_sequences, test_size=0.2, random_state=42
            )

            # Scale the features
            X_train_reshaped = X_train.reshape(-1, n_features)
            X_test_reshaped = X_test.reshape(-1, n_features)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_reshaped)
            X_test_scaled = scaler.transform(X_test_reshaped)

            X_train = X_train_scaled.reshape(X_train.shape)
            X_test = X_test_scaled.reshape(X_test.shape)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def add_team_form_features(self, df):
        """Add team form features based on recent performance."""
        try:
            # Initialize form features
            df["team1_form"] = 0.0
            df["team2_form"] = 0.0
            df["team1_recent_wins"] = 0
            df["team2_recent_wins"] = 0
            df["team1_recent_score"] = 0.0
            df["team2_recent_score"] = 0.0

            # Sort by date
            df["date"] = pd.to_datetime(df[["year", "month", "day"]])
            df = df.sort_values("date")

            # Calculate rolling statistics for each team
            for idx, row in df.iterrows():
                # Get previous matches
                prev_matches = df.loc[: idx - 1]
                if len(prev_matches) > 0:
                    # Team 1 form
                    team1_matches = prev_matches[
                        (prev_matches["team1"] == row["team1"])
                        | (prev_matches["team2"] == row["team1"])
                    ].tail(
                        5
                    )  # Last 5 matches

                    if len(team1_matches) > 0:
                        # Calculate form (weighted average of results)
                        weights = np.array([0.4, 0.3, 0.15, 0.1, 0.05])[
                            : len(team1_matches)
                        ]
                        weights = weights / weights.sum()  # Normalize weights

                        team1_results = []
                        team1_scores = []
                        for _, match in team1_matches.iterrows():
                            if match["team1"] == row["team1"]:
                                team1_results.append(1 if match["winner"] == 1 else 0)
                                team1_scores.append(match["team1_score"])
                            else:
                                team1_results.append(1 if match["winner"] == 0 else 0)
                                team1_scores.append(match["team2_score"])

                        df.at[idx, "team1_form"] = np.average(
                            team1_results, weights=weights
                        )
                        df.at[idx, "team1_recent_wins"] = sum(team1_results)
                        df.at[idx, "team1_recent_score"] = np.mean(team1_scores)

                    # Team 2 form
                    team2_matches = prev_matches[
                        (prev_matches["team1"] == row["team2"])
                        | (prev_matches["team2"] == row["team2"])
                    ].tail(
                        5
                    )  # Last 5 matches

                    if len(team2_matches) > 0:
                        # Calculate form (weighted average of results)
                        weights = np.array([0.4, 0.3, 0.15, 0.1, 0.05])[
                            : len(team2_matches)
                        ]
                        weights = weights / weights.sum()  # Normalize weights

                        team2_results = []
                        team2_scores = []
                        for _, match in team2_matches.iterrows():
                            if match["team1"] == row["team2"]:
                                team2_results.append(1 if match["winner"] == 1 else 0)
                                team2_scores.append(match["team1_score"])
                            else:
                                team2_results.append(1 if match["winner"] == 0 else 0)
                                team2_scores.append(match["team2_score"])

                        df.at[idx, "team2_form"] = np.average(
                            team2_results, weights=weights
                        )
                        df.at[idx, "team2_recent_wins"] = sum(team2_results)
                        df.at[idx, "team2_recent_score"] = np.mean(team2_scores)

            return df

        except Exception as e:
            logger.error(f"Error in add_team_form_features: {str(e)}")
            return df

    def create_sequences(self, df):
        """Create sequences for time series prediction with enhanced features."""
        try:
            # Drop non-numeric columns
            df = df.select_dtypes(include=[np.number])

            # Separate features and target
            X = df.drop(["winner"], axis=1, errors="ignore")
            y = df["winner"]

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Create sequences with additional features
            X_seq, y_seq = [], []
            for i in range(len(X_scaled) - self.sequence_length):
                # Get current sequence
                current_sequence = X_scaled[i : (i + self.sequence_length)]

                # Calculate sequence statistics
                sequence_stats = np.array(
                    [
                        np.nanmean(current_sequence, axis=0),  # Mean of each feature
                        np.nanstd(current_sequence, axis=0),  # Std of each feature
                        np.nanmax(current_sequence, axis=0),  # Max of each feature
                        np.nanmin(current_sequence, axis=0),  # Min of each feature
                        np.nanmedian(
                            current_sequence, axis=0
                        ),  # Median of each feature
                        np.nanpercentile(
                            current_sequence, 25, axis=0
                        ),  # 25th percentile
                        np.nanpercentile(
                            current_sequence, 75, axis=0
                        ),  # 75th percentile
                    ]
                )

                # Calculate trend features
                trends = np.array(
                    [
                        np.nanmean(
                            np.diff(current_sequence, axis=0), axis=0
                        ),  # Average change
                        np.nansum(np.diff(current_sequence, axis=0) > 0, axis=0)
                        / (self.sequence_length - 1),  # Proportion of increases
                        np.nan_to_num(
                            np.corrcoef(
                                np.arange(self.sequence_length),
                                current_sequence,
                                rowvar=False,
                            )[0, 1:]
                        ),  # Correlation with time
                    ]
                )

                # Calculate volatility features
                volatility = np.array(
                    [
                        np.nanstd(
                            np.diff(current_sequence, axis=0), axis=0
                        ),  # Volatility of changes
                        np.nanmax(
                            np.abs(np.diff(current_sequence, axis=0)), axis=0
                        ),  # Maximum absolute change
                    ]
                )

                # Flatten and combine all features
                sequence_features = np.concatenate(
                    [
                        current_sequence.flatten(),  # Original sequence
                        sequence_stats.flatten(),  # Sequence statistics
                        trends.flatten(),  # Trend features
                        volatility.flatten(),  # Volatility features
                    ]
                )

                X_seq.append(sequence_features)
                y_seq.append(y.iloc[i + self.sequence_length])

            return np.array(X_seq), np.array(y_seq)

        except Exception as e:
            logger.error(f"Error in create_sequences: {str(e)}")
            raise

    def train_and_evaluate(self):
        """Train and evaluate the time series model."""
        try:
            # Load and preprocess data
            X_train, X_test, y_train, y_test = self.load_data()

            # Create datasets and dataloaders
            train_dataset = IPLDataset(X_train, y_train)
            test_dataset = IPLDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Create model
            input_size = X_train.shape[2]  # Number of features
            model = IPLLSTM(input_size=input_size).to(self.device)

            # Define loss function and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5
            )

            # Training loop
            n_epochs = 100
            best_val_loss = float("inf")
            patience = 20
            patience_counter = 0
            train_losses = []
            val_losses = []

            for epoch in range(n_epochs):
                model.train()
                total_train_loss = 0

                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item()

                # Validation
                model.eval()
                total_val_loss = 0
                val_predictions = []
                val_targets = []

                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y.unsqueeze(1))
                        total_val_loss += loss.item()

                        val_predictions.extend((outputs > 0.5).cpu().numpy())
                        val_targets.extend(batch_y.cpu().numpy())

                avg_train_loss = total_train_loss / len(train_loader)
                avg_val_loss = total_val_loss / len(test_loader)
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)

                # Learning rate scheduling
                scheduler.step(avg_val_loss)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), "models/time_series_best_model.pth")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                    )

            # Plot training history
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label="Training Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.title("Model Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

            plt.tight_layout()
            plt.savefig("models/time_series_training_history.png")
            plt.close()

            # Load best model for evaluation
            model.load_state_dict(torch.load("models/time_series_best_model.pth"))
            model.eval()

            # Final evaluation
            y_pred = []
            with torch.no_grad():
                for batch_X, _ in test_loader:
                    batch_X = batch_X.to(self.device)
                    outputs = model(batch_X)
                    y_pred.extend((outputs > 0.5).cpu().numpy())

            y_pred = np.array(y_pred).flatten()
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.savefig("models/time_series_confusion_matrix.png")
            plt.close()

            # Save preprocessors
            joblib.dump(self.scaler, "models/time_series_scaler.joblib")
            joblib.dump(self.imputer, "models/time_series_imputer.joblib")

            results = {
                "accuracy": accuracy,
                "classification_report": report,
                "loss": avg_val_loss,  # Add the loss to the results
            }

            return results

        except Exception as e:
            logger.error(f"Error in train_and_evaluate: {str(e)}")
            raise


if __name__ == "__main__":
    model = IPLTimeSeriesModel()
    model.train_and_evaluate()
