import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(model_files):
    """Load CSV files for each model evaluation."""
    return [pd.read_csv(file) for file in model_files]


def calculate_custom_scores(df):
    """
    Calculate score1 and score2 based on the specified criteria.

    score1: Average of q1-q15 and conditionally q_translation, normalized to 0-100%
    score2: score1 with length penalty applied when length_score < 5
    """
    # Get all question columns (q1 through q15)
    q_cols = [col for col in df.columns if col.startswith("q") and col[1:].isdigit()]

    # First average: q1-q15 and conditionally q_translation
    df["score1_raw"] = df.apply(
        lambda row: np.mean(
            [row[col] for col in q_cols]
            + ([row["q_translation"]] if row["language"] != "en" else [])
        ),
        axis=1,
    )

    # Normalize score1 to 0-100% (assuming scores are between 1-5)
    df["score1"] = (df["score1_raw"] - 1) / 4 * 100

    # Second score: apply length penalty when length_score < 5
    df["score2"] = df.apply(
        lambda row: row["score1"] * (row["length_score"] / 5)
        if row["length_score"] < 5
        else row["score1"],
        axis=1,
    )

    return df


def calculate_average_ranks(evaluator_dfs, score_column):
    """Calculate the average rank of each model across evaluators."""
    # Calculate mean score for each model within each evaluator
    model_rankings = []

    for df in evaluator_dfs:
        rankings = df.groupby("model_name")[score_column].mean().reset_index()
        model_rankings.append(rankings)

    # Combine rankings from all evaluators
    all_rankings = pd.concat(model_rankings)
    avg_rankings = (
        all_rankings.groupby("model_name")[score_column]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Sort by mean score
    avg_rankings = avg_rankings.sort_values("mean", ascending=False)

    return avg_rankings


def calculate_spearman_correlation(evaluator_ranks, score_column):
    """Calculate average Spearman correlation between evaluators."""
    correlations = []

    for i in range(len(evaluator_ranks)):
        for j in range(i + 1, len(evaluator_ranks)):
            df1 = evaluator_ranks[i].set_index("model_name")
            df2 = evaluator_ranks[j].set_index("model_name")

            # Join the rankings
            joined = df1.join(df2, lsuffix="_1", rsuffix="_2")

            # Calculate Spearman correlation
            corr = (
                joined[f"{score_column}_1"]
                .rank(ascending=False)
                .corr(
                    joined[f"{score_column}_2"].rank(ascending=False), method="spearman"
                )
            )
            correlations.append(corr)

    return np.mean(correlations), np.std(correlations)


def get_readable_model_name(model_name):
    """Convert the model name to a more readable format."""
    # Define mapping for model name cleanup
    model_mapping = {
        "solidrust/Gemma-2-Ataraxy-9B-AWQ": "Gemma 2 Ataraxy (9B)",
        "harborwater/Gemma-2-9B-It-SPPO-Iter3-AWQ": "Gemma 2 SPPO (9B)",
        "Orion-zhen/aya-expanse-32b-AWQ": "Aya Expanse (32B)",
        "solidrust/gemma-2-9b-it-AWQ": "Gemma 2 (9B)",
        "casperhansen/mistral-nemo-instruct-2407-awq": "Mistral NeMo (12B)",
        "Orion-zhen/aya-expanse-8b-AWQ": "Aya Expanse (8B)",
        "arcee-ai/Arcee-Blitz-AWQ": "Arcee Blitz (24B)",
        "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ": "Mistral Small (24B)",
        "AMead10/SuperNova-Medius-AWQ": "SuperNova Medius (14B)",
        "Qwen/Qwen2.5-14B-Instruct-AWQ": "Qwen 2.5 (14B)",
        "AMead10/c4ai-command-r-08-2024-awq": "C4AI Command R (32B)",
        "alijawad07/aya-23-8B-AWQ-GEMM": "Aya 23 (8B)",
        "solidrust/Hermes-3-Llama-3.1-8B-AWQ": "Hermes 3 Llama 3.1 (8B)",
        "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4": "Llama 3.1 (8B)",
        "stelterlab/phi-4-AWQ": "Phi-4 (14B)",
    }

    # Return the mapped name if it exists, otherwise return the original
    return model_mapping.get(model_name, model_name)


def plot_rankings_subplots(avg_score1, avg_score2):
    """Create two subplots - one for score1 (short text) and one for score2 (long text)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Apply readable model names
    avg_score1["display_name"] = avg_score1["model_name"].apply(get_readable_model_name)
    avg_score2["display_name"] = avg_score2["model_name"].apply(get_readable_model_name)

    # Plot for Short Text Writing (score1)
    bars1 = ax1.bar(
        avg_score1["display_name"],
        avg_score1["mean"],
        yerr=avg_score1["std"],
        capsize=4,
        color="cornflowerblue",
    )

    # Add rank numbers for score1
    for i, (_, row) in enumerate(avg_score1.iterrows()):
        ax1.text(i, row["mean"] + 2, f"#{i+1}", ha="center", fontweight="bold")

    ax1.set_title("Writing (short texts)", fontsize=16)
    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("Score (%)", fontsize=12)

    # Set X tick labels with rotation
    ax1.set_xticks(range(len(avg_score1)))
    ax1.set_xticklabels(avg_score1["display_name"], rotation=45, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Plot for Long Text Writing with Length Penalty (score2)
    bars2 = ax2.bar(
        avg_score2["display_name"],
        avg_score2["mean"],
        yerr=avg_score2["std"],
        capsize=4,
        color="mediumseagreen",
    )

    # Add rank numbers for score2
    for i, (_, row) in enumerate(avg_score2.iterrows()):
        ax2.text(i, row["mean"] + 2, f"#{i+1}", ha="center", fontweight="bold")

    ax2.set_title("Writing (long texts)", fontsize=16)
    ax2.set_xlabel("Model", fontsize=12)
    ax2.set_ylabel("Score (%)", fontsize=12)

    # Set X tick labels with rotation
    ax2.set_xticks(range(len(avg_score2)))
    ax2.set_xticklabels(avg_score2["display_name"], rotation=45, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main function for model evaluation."""
    sns.set()

    # List of model files
    model_files = ["gemma2.csv", "mistral.csv", "llama.csv", "aya_expanse.csv"]

    # Load and process data
    print(f"Loading data from {model_files}...")
    dataframes = load_data(model_files)
    processed_dfs = [calculate_custom_scores(df) for df in dataframes]

    # Get individual model rankings for each evaluator
    score1_rankings = []
    score2_rankings = []

    for df in processed_dfs:
        # Score1 rankings
        s1_rank = df.groupby("model_name")["score1"].mean().reset_index()
        score1_rankings.append(s1_rank)

        # Score2 rankings
        s2_rank = df.groupby("model_name")["score2"].mean().reset_index()
        score2_rankings.append(s2_rank)

    # Calculate average rankings across evaluators
    avg_score1_rankings = calculate_average_ranks(processed_dfs, "score1")
    avg_score2_rankings = calculate_average_ranks(processed_dfs, "score2")

    # Calculate Spearman correlations
    s1_corr_mean, s1_corr_std = calculate_spearman_correlation(
        score1_rankings, "score1"
    )
    s2_corr_mean, s2_corr_std = calculate_spearman_correlation(
        score2_rankings, "score2"
    )

    # Print results with percentages
    print("\nAverage model rankings (Writing):")
    for i, (_, row) in enumerate(avg_score1_rankings.iterrows()):
        readable_name = get_readable_model_name(row["model_name"])
        print(f"#{i+1}: {readable_name:<30} {row['mean']:.1f}% ± {row['std']:.1f}%")

    print("\nAverage model rankings (Writing with Length Penalty):")
    for i, (_, row) in enumerate(avg_score2_rankings.iterrows()):
        readable_name = get_readable_model_name(row["model_name"])
        print(f"#{i+1}: {readable_name:<30} {row['mean']:.1f}% ± {row['std']:.1f}%")

    print(
        f"\nAverage Spearman correlation for Short Text: {s1_corr_mean:.3f} ± {s1_corr_std:.3f}"
    )
    print(
        f"Average Spearman correlation for Long Text: {s2_corr_mean:.3f} ± {s2_corr_std:.3f}"
    )

    # Create and save plot with two subplots
    fig = plot_rankings_subplots(avg_score1_rankings, avg_score2_rankings)
    plt.savefig(
        "model_rankings_comparison.jpg", dpi=300, pad_inches=0, bbox_inches="tight"
    )
    print("\nPlot saved as 'model_rankings_comparison.jpg'")

    plt.show()


if __name__ == "__main__":
    main()
