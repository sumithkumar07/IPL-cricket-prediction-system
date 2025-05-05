"""
Training script for IPL match prediction models.
"""

import logging
import matplotlib.pyplot as plt
from ml_model.ensemble import IPLEnsembleModel
from ml_model.time_series import IPLTimeSeriesModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_model_comparison(ensemble_results, time_series_results):
    """Plot accuracy comparison of different models."""
    try:
        # Extract accuracies
        accuracies = [
            ensemble_results["winner_results"]["xgboost"]["accuracy"],
            ensemble_results["winner_results"]["random_forest"]["accuracy"],
            ensemble_results["winner_results"]["neural_network"]["accuracy"],
            ensemble_results["winner_results"]["voting"]["accuracy"],
            time_series_results["accuracy"] if time_series_results else 0,
        ]

        # Model names
        models = [
            "XGBoost",
            "Random Forest",
            "Neural Network",
            "Voting Classifier",
            "Time Series",
        ]

        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, accuracies)

        # Customize plot
        plt.title("Model Comparison - IPL Match Prediction", fontsize=14, pad=20)
        plt.xlabel("Models", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
            )

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig("models/model_comparison.png")
        plt.close()

    except Exception as e:
        logger.error(f"Error in plot_model_comparison: {str(e)}")
        raise


def main():
    """Main function to train and evaluate models."""
    try:
        # Train ensemble model
        logger.info("\nTraining ensemble model...")
        ensemble_model = IPLEnsembleModel()
        ensemble_results = ensemble_model.train_and_evaluate()

        # Print ensemble model results
        logger.info("\nEnsemble Model Results:")
        logger.info("\nWinner Prediction:")
        for model_name, metrics in ensemble_results["winner_results"].items():
            logger.info(f"\n{model_name.upper()} Accuracy: {metrics['accuracy']:.4f}")

        logger.info("\nScore Prediction:")
        for team in ["team1", "team2"]:
            logger.info(f"\n{team.upper()} Scores:")
            for model_name, metrics in ensemble_results["score_results"][team].items():
                logger.info(f"\n{model_name.upper()}:")
                logger.info(f"MAE: {metrics['mae']:.4f}")
                logger.info(f"RMSE: {metrics['rmse']:.4f}")
                logger.info(f"R² Score: {metrics['r2']:.4f}")

        # Train time series model
        logger.info("\nTraining time series model...")
        time_series_model = IPLTimeSeriesModel()
        time_series_results = time_series_model.train_and_evaluate()

        if time_series_results:
            logger.info("\nTime Series Model Results:")
            logger.info(f"Accuracy: {time_series_results['accuracy']:.4f}")
            if "loss" in time_series_results:
                logger.info(f"Loss: {time_series_results['loss']:.4f}")
            logger.info("\nClassification Report:")
            logger.info(time_series_results["classification_report"])

        # Plot model comparison
        plot_model_comparison(ensemble_results, time_series_results)

        # Find best model
        best_model = max(
            [
                (name, metrics["accuracy"])
                for name, metrics in ensemble_results["winner_results"].items()
            ],
            key=lambda x: x[1],
        )
        logger.info(
            f"\nBest Model: {best_model[0].upper()} (Accuracy: {best_model[1]:.4f})"
        )

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
