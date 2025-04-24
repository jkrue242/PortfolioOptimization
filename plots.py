import matplotlib.pyplot as plt


def plot_features(data):
    feature_names = data.columns.get_level_values(1).unique()

    for feature_name in feature_names:
        # Feature cross sections
        feature_data = data.xs(feature_name, level=1, axis=1)

        plt.figure(figsize=(14, 7))
        ax = plt.gca()

        feature_data.plot(ax=ax)

        plt.title(f"{feature_name} Over Time for All ETFs")
        plt.xlabel("Date")
        plt.ylabel(feature_name.replace("_", " ").title())

        plt.legend(title="ETFs")

        plt.grid(True)
        plt.show()


def plot_training_history(history):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot([-l for l in history["train_losses"]], label="Train Loss")
    plt.plot([-l for l in history["val_losses"]], label="Validation Loss")
    plt.title("Sharpe Ratio Loss (higher is better)")
    plt.xlabel("Epoch")
    plt.ylabel("Sharpe Ratio")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(history["val_sharpes"], label="Validation Sharpe")
    plt.title("Validation Sharpe Ratio")
    plt.xlabel("Epoch")
    plt.ylabel("Sharpe Ratio")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_test_results(results, test_data, etfs):
    dates = results["dates"]
    cumulative_return_feature_data = test_data.loc[dates].xs(
        "cumulative_return", level=1, axis=1
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

    axes[0].set_title("Portfolio Weights Over Time (Model Predictions on Test Set)")
    axes[0].set_ylabel("Weight")
    for i, etf in enumerate(etfs):
        axes[0].plot(dates, results["weights"][: len(dates), i], label=etf)
    axes[0].legend(title="ETFs")
    axes[0].grid(True)

    axes[1].set_title("Cumulative Return Feature Over Time for Each ETF (Test Set)")
    axes[1].set_ylabel("Cumulative Return")
    cumulative_return_feature_data.plot(ax=axes[1])
    axes[1].legend(title="ETFs")
    axes[1].grid(True)

    axes[2].set_title("Cumulative Sharpe Ratio of Model Portfolio Over Time (Test Set)")
    axes[2].set_ylabel("Sharpe Ratio")
    axes[2].set_xlabel("Date")
    axes[2].plot(
        dates[: len(results["cumulative_sharpe_ratios"])],
        results["cumulative_sharpe_ratios"][: len(dates)],
    )
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
