import pandas as pd
import os


def rank_stories_by_prompt_id(csv_file, target_prompt_id, show_quantiles=False):
    """
    Load a CSV file, filter stories by prompt_id, verify prompt text consistency,
    and rank stories from best to worst based on overall_score.

    Args:
        csv_file (str): Path to the CSV file
        target_prompt_id (str): The prompt ID to filter by
        show_quantiles (bool): If True, show only stories at 0.25, 0.5, 0.75, and 1.0 quantiles
    """
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} does not exist.")
        return

    # Load the CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded {csv_file} with {len(df)} rows")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Filter stories by prompt_id
    filtered_df = df[df["prompt_id"] == target_prompt_id]

    if filtered_df.empty:
        print(f"No stories found with prompt_id {target_prompt_id}")
        return

    print(f"Found {len(filtered_df)} stories with prompt_id {target_prompt_id}")

    # Verify prompt text consistency
    prompt_texts = filtered_df["prompt"].unique()

    print("\n=== PROMPT TEXT VERIFICATION ===")
    if len(prompt_texts) == 1:
        print("✓ Prompt text is consistent across all stories.")
        prompt_text = prompt_texts[0]
    else:
        print("⚠ Found different prompt texts for the same prompt_id:")
        for i, text in enumerate(prompt_texts, 1):
            print(f"Variation {i}: {text}")
        prompt_text = prompt_texts[0]

    print("\n=== PROMPT TEXT ===")
    print(prompt_text)

    # Sort by overall_score in descending order (best to worst)
    ranked_df = filtered_df.sort_values(by="overall_score", ascending=False)

    if show_quantiles:
        # Get number of stories
        n_stories = len(ranked_df)
        print(f"\n=== SHOWING 4 REPRESENTATIVE STORIES (TOTAL: {n_stories}) ===")
        print(
            "Stories shown are at the following quantiles: 1.0 (best), 0.75, 0.5 (median), 0.25"
        )

        # Calculate indices for quantiles
        best_idx = 0  # Best story (1.0 quantile)
        q75_idx = int(n_stories * 0.25) - 1  # 0.75 quantile
        median_idx = int(n_stories * 0.5) - 1  # 0.5 quantile
        q25_idx = int(n_stories * 0.75) - 1  # 0.25 quantile

        # If we have fewer than 4 stories, just show all of them
        indices_to_show = sorted(list(set([best_idx, q75_idx, median_idx, q25_idx])))
        indices_to_show = [idx for idx in indices_to_show if 0 <= idx < n_stories]

        # Dict to map index to quantile description
        quantile_labels = {
            best_idx: "1.0 (BEST)",
            q75_idx: "0.75",
            median_idx: "0.5 (MEDIAN)",
            q25_idx: "0.25",
        }
    else:
        print(f"\n=== STORIES RANKED FROM BEST TO WORST (TOTAL: {len(ranked_df)}) ===")
        indices_to_show = range(len(ranked_df))
        quantile_labels = {idx: str(idx + 1) for idx in indices_to_show}

    # Display the selected stories
    for idx in indices_to_show:
        row = ranked_df.iloc[idx]
        quantile_label = quantile_labels.get(idx, str(idx + 1))

        print(f"\n----- STORY #{quantile_label} -----")
        print(f"Rank: {idx+1} of {len(ranked_df)}")
        print(f"Model: {row['model_name']}")
        print(f"Overall Score: {row['overall_score']}")
        print(f"Length Score: {row['length_score']}")

        # Create a scores summary
        scores = [f"q{i+1}: {row[f'q{i+1}']}" for i in range(15)]
        scores.append(f"q_translation: {row['q_translation']}")

        print("Scores:", ", ".join(scores))

        print("\nSTORY TEXT:")
        print(row["story_text"])
        print("-" * 80)


def main():
    # Default file path and prompt ID
    csv_file = "average.csv"
    target_prompt_id = "p6d7d0bcdf6"

    print("=== STORY RANKING SCRIPT ===")

    # Allow custom file input
    custom_file = input(
        f"Enter CSV file path (or press Enter to use default '{csv_file}'): "
    )
    if custom_file.strip():
        csv_file = custom_file

    # Allow custom prompt ID input
    custom_prompt_id = input(
        f"Enter prompt ID (or press Enter to use default '{target_prompt_id}'): "
    )
    if custom_prompt_id.strip():
        target_prompt_id = custom_prompt_id

    # Ask about showing quantiles
    show_quantiles = input(
        "Show only 4 representative stories based on quantiles? (y/n, default: y): "
    ).lower()
    show_quantiles = show_quantiles != "n"  # Default to True unless explicitly 'n'

    # Run the ranking function
    rank_stories_by_prompt_id(csv_file, target_prompt_id, show_quantiles)


if __name__ == "__main__":
    main()
