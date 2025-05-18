from config import ConfigManager
from model import MyNetwork  # Assuming you have a `MyNetwork` class with a `.fit()` method

def main():
    # Get the configuration and arguments
    config = ConfigManager.get_config_and_arguments()

    # Instantiate the model using the configuration
    model = MyNetwork(config)

    # Train the model
    train_losses, n_epochs_trained, _ = model.fit(train_loader)

    # Save the trained model
    model.save_model(config.output_path)

    # Predict on the test set
    predictions = model.predict(test_loader)

    # Generate results (plots, metrics, etc.) based on predictions
    model.generate_results(predictions, config.output_path)

if __name__ == "__main__":
    main()
