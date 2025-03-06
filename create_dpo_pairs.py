import pandas as pd
import json
from datasets import load_dataset
from fast_langdetect import detect
from tqdm import tqdm


def create_enhanced_dpo_pairs(df, min_score_diff=0.4, confidence_threshold=0.8):
    """
    Creates DPO pairs using both language verification and quality scores.

    Args:
        df: DataFrame containing stories with language tags
        min_score_diff: Minimum quality score difference for quality-based pairs
        confidence_threshold: Minimum confidence for language detection

    Returns:
        List of DPO pairs in the format {prompt, chosen, rejected}
    """
    print("Starting enhanced DPO pair creation...")
    pairs = []
    used_stories = set()

    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Add columns for language verification
    df_copy["detected_lang"] = None
    df_copy["lang_confidence"] = 0.0
    df_copy["language_correct"] = False

    print("Detecting language for all stories...")
    # Process each story to detect its language
    for idx, row in tqdm(df_copy.iterrows(), total=len(df_copy)):
        if pd.notna(row["story_text"]) and len(row["story_text"].strip()) > 20:
            try:
                # Clean the text (remove newlines as recommended in the docs)
                text = row["story_text"].replace("\n", " ")

                # Detect language with higher accuracy model
                result = detect(text, low_memory=False)

                df_copy.at[idx, "detected_lang"] = result["lang"]
                df_copy.at[idx, "lang_confidence"] = result["score"]

                # Check if detected language matches expected language
                expected_lang = row["language"].lower()
                detected_lang = result["lang"].lower()

                # Mark as correct if languages match and detection is confident
                if (
                    expected_lang == detected_lang
                    and result["score"] >= confidence_threshold
                ):
                    df_copy.at[idx, "language_correct"] = True
            except Exception as e:
                print(f"Error detecting language for story {idx}: {e}")
                # Continue processing other stories

    print("Creating language-based pairs...")
    # First, create language-based pairs
    for prompt_id, prompt_group in df_copy.groupby("prompt_id"):
        if len(prompt_group) < 2:
            continue

        # Find stories with correct and incorrect language
        correct_lang_stories = prompt_group[prompt_group["language_correct"] == True]
        incorrect_lang_stories = prompt_group[prompt_group["language_correct"] == False]

        # Skip if we don't have both correct and incorrect stories
        if correct_lang_stories.empty or incorrect_lang_stories.empty:
            continue

        # Create pairs matching correct language stories with incorrect ones
        for _, correct_row in correct_lang_stories.iterrows():
            # Skip if this story is already used
            if correct_row["story_text"] in used_stories:
                continue

            for _, incorrect_row in incorrect_lang_stories.iterrows():
                # Skip if this story is already used
                if incorrect_row["story_text"] in used_stories:
                    continue

                # Create a language-based pair
                pairs.append(
                    {
                        "prompt": correct_row["prompt"],
                        "chosen": correct_row["story_text"],
                        "rejected": incorrect_row["story_text"],
                        "pair_type": "language",  # Optional metadata
                    }
                )

                # Mark stories as used
                used_stories.add(correct_row["story_text"])
                used_stories.add(incorrect_row["story_text"])
                break  # Move to next correct story

    print("Creating quality-based pairs...")
    # Then, create quality-based pairs for remaining stories
    all_quality_pairs = []

    for prompt_id, prompt_group in df_copy.groupby("prompt_id"):
        if len(prompt_group) < 2:
            continue

        # Find all possible quality pairs for this prompt
        for i, row_i in prompt_group.iterrows():
            # Skip if already used
            if row_i["story_text"] in used_stories:
                continue

            for j, row_j in prompt_group.iterrows():
                # Skip if same story or already used
                if i == j or row_j["story_text"] in used_stories:
                    continue

                # Calculate quality score difference using the specified metrics
                s1 = (row_i["q11"] + row_i["q12"] + row_i["q14"] + row_i["q1"]) / 4
                s2 = (row_j["q11"] + row_j["q12"] + row_j["q14"] + row_j["q1"]) / 4
                score_diff = s1 - s2

                # Only consider pairs where both stories have correct language
                if (
                    score_diff >= min_score_diff
                    and row_i["language_correct"] == True
                    and row_j["language_correct"] == True
                ):
                    all_quality_pairs.append(
                        {
                            "preferred_idx": i,
                            "rejected_idx": j,
                            "prompt": row_i["prompt"],
                            "preferred": row_i["story_text"],
                            "rejected": row_j["story_text"],
                            "score_diff": float(score_diff),
                            "pair_type": "quality",
                        }
                    )

    # Sort quality pairs by score difference (largest first)
    all_quality_pairs.sort(key=lambda x: x["score_diff"], reverse=True)

    print("Selecting final quality pairs...")
    # Greedily select quality pairs
    for pair in all_quality_pairs:
        if pair["preferred"] in used_stories or pair["rejected"] in used_stories:
            continue

        # Add this quality pair
        pairs.append(
            {
                "prompt": pair["prompt"],
                "chosen": pair["preferred"],
                "rejected": pair["rejected"],
                "pair_type": "quality",  # Optional metadata
            }
        )

        # Mark these stories as used
        used_stories.add(pair["preferred"])
        used_stories.add(pair["rejected"])

    print(f"Created {len(pairs)} DPO pairs total.")
    print(
        f"  - Language-based pairs: {len([p for p in pairs if p.get('pair_type') == 'language'])}"
    )
    print(
        f"  - Quality-based pairs: {len([p for p in pairs if p.get('pair_type') == 'quality'])}"
    )

    # Remove the pair_type field before returning (unless needed for analysis)
    for pair in pairs:
        if "pair_type" in pair:
            del pair["pair_type"]

    return pairs


def generate_dpo_dataset(dataset_name, output_file, min_score_diff=0.4):
    """
    Generates a DPO dataset from a Hugging Face dataset and saves it as JSONL.

    Args:
        dataset_name: Name of the Hugging Face dataset
        data_files: Dictionary of file names to load from the dataset
        output_file: Path to output JSONL file
        min_score_diff: Minimum score difference for quality-based pairs
    """
    # Load the dataset from Hugging Face
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name)

    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(dataset["train"])

    # Create DPO pairs
    pairs = create_enhanced_dpo_pairs(df, min_score_diff=min_score_diff)

    # Write pairs to JSONL file
    print(f"Writing {len(pairs)} pairs to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Dataset saved as '{output_file}' in chatml.prompt_pairs format")


# Usage example
if __name__ == "__main__":
    # Load from the Hugging Face dataset
    generate_dpo_dataset(
        dataset_name="lars1234/story_writing_benchmark",
        output_file="story_dpo_pairs.jsonl",
        min_score_diff=0.4,
    )
