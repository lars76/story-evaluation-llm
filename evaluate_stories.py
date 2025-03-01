import pandas as pd
import argparse
import re
import csv
from typing import List
import time
import logging
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("story_evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger("story_evaluator")

# Default constants
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MIN_P = 0.0
DEFAULT_MODEL = "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_NUM_GPUS = 4

# Evaluation criteria with examples for 1, 3, and 5 ratings
EVALUATION_CRITERIA = [
    # Technical Quality
    {
        "question": "Is the text free from errors in grammar, spelling, and punctuation?",
        "examples": {
            "1": "Multiple errors per paragraph that impede understanding",
            "3": "Occasional minor errors that don't affect comprehension",
            "5": "Clean, professional-level text with virtually no errors",
        },
    },
    {
        "question": "Is the writing clear and understandable?",
        "examples": {
            "1": "Confusing sentences and unclear descriptions that make the story hard to follow",
            "3": "Generally clear but with some passages that require rereading",
            "5": "Consistently clear prose that conveys ideas precisely",
        },
    },
    # Narrative Elements
    {
        "question": "Do events and ideas connect logically from one to the next?",
        "examples": {
            "1": "Plot points appear randomly with no clear causation between events",
            "3": "Story generally flows but has some unexplained jumps or connections",
            "5": "Each event naturally follows from what came before in a satisfying chain",
        },
    },
    {
        "question": "Are scenes well-constructed with clear purpose?",
        "examples": {
            "1": "Scenes drag on without advancing plot or character development",
            "3": "Most scenes serve a purpose but some feel unnecessary",
            "5": "Every scene efficiently moves the story forward or develops characters",
        },
    },
    {
        "question": "Does the text make sense within its own context?",
        "examples": {
            "1": "Contains contradictions or elements that break the story's internal logic",
            "3": "Generally consistent with a few minor logical issues",
            "5": "Creates a coherent world with consistent internal rules",
        },
    },
    # Character Presentation
    {
        "question": "Are characters presented consistently?",
        "examples": {
            "1": "Characters behave differently from scene to scene without explanation",
            "3": "Characters are mostly consistent with occasional unexplained behavior",
            "5": "Characters act consistently with their established traits throughout",
        },
    },
    {
        "question": "Do characters act in ways that make sense in context?",
        "examples": {
            "1": "Characters make decisions purely to advance the plot rather than from motivation",
            "3": "Characters' actions generally make sense but sometimes lack clear motivation",
            "5": "Characters' actions always arise naturally from their personalities and situations",
        },
    },
    # Writing Effectiveness
    {
        "question": "Does the text avoid monotonous sentence patterns?",
        "examples": {
            "1": "Repetitive sentence structures (e.g., 'He did X. He did Y. He felt Z.')",
            "3": "Some variety in sentence structure but with noticeable patterns",
            "5": "Rich variety of sentence structures that create rhythm and flow",
        },
    },
    {
        "question": "Does the text avoid clichés and overused phrases?",
        "examples": {
            "1": "Relies heavily on phrases like 'sharp mind, sharper eyes' or 'melting pot of glamour and misery'",
            "3": "Some original language mixed with occasional clichés",
            "5": "Consistently fresh language with creative metaphors and descriptions",
        },
    },
    {
        "question": "Does dialogue sound natural for the characters?",
        "examples": {
            "1": "All characters speak in the same voice with formal, stiff phrasing",
            "3": "Dialogue generally works but doesn't always match the character's background",
            "5": "Each character has a distinct voice that reflects their personality and background",
        },
    },
    # Originality and Depth
    {
        "question": "Does the text avoid predictable narrative tropes?",
        "examples": {
            "1": "Story follows genre formulas exactly with no subversion or fresh elements",
            "3": "Uses some common tropes but attempts to add original elements",
            "5": "Takes creative approaches to storytelling that surprise the reader",
        },
    },
    {
        "question": "Are characters portrayed with depth rather than as one-dimensional?",
        "examples": {
            "1": "Flat stereotypes (the cold widow, the mysterious lover, the tough detective)",
            "3": "Characters with some personality but predictable motivations",
            "5": "Complex characters with realistic contradictions and development",
        },
    },
    {
        "question": "Do character interactions feel realistic and earned?",
        "examples": {
            "1": "Relationships develop instantly with no buildup or foundation",
            "3": "Relationships generally convincing but sometimes progress too quickly",
            "5": "Relationships develop organically through meaningful interactions",
        },
    },
    # Reader Experience
    {
        "question": "Does the text hold interest?",
        "examples": {
            "1": "Story is tedious with predictable outcomes and no stakes",
            "3": "Has engaging moments but also sections where interest wanes",
            "5": "Consistently engages the reader with tension, curiosity, or emotional investment",
        },
    },
    {
        "question": "Does the plot resolution feel earned and satisfying?",
        "examples": {
            "1": "Villain confesses suddenly with minimal investigation or stakes",
            "3": "Resolution makes sense but lacks surprise or emotional impact",
            "5": "Clever resolution that builds on established clues while surprising the reader",
        },
    },
]

# Special question for non-English texts
TRANSLATION_QUESTION = {
    "question": "Does this appear to be a natural text rather than a translation?",
    "examples": {
        "1": "Contains awkward phrasing that sounds like direct translation from another language",
        "3": "Generally natural with occasional phrases that feel translated",
        "5": "Reads like it was originally written in this language with natural idioms and flow",
    },
}


def word_count(text: str) -> int:
    """Count the number of words in a text."""
    return len(re.findall(r"\w+", text))


def calculate_length_score(
    text: str, target_word_count: int, tolerance: int = 200
) -> int:
    """
    Calculate a score from 1-5 based on how close the text is to the target length.
    5: Within tolerance
    4: Up to 1.5x tolerance difference
    3: Up to 2x tolerance difference
    2: Up to 3x tolerance difference
    1: More than 3x tolerance difference
    """
    text_length = word_count(text)
    difference = abs(text_length - target_word_count)

    if difference <= tolerance:
        return 5
    elif difference <= 1.5 * tolerance:
        return 4
    elif difference <= 2 * tolerance:
        return 3
    elif difference <= 3 * tolerance:
        return 2
    else:
        return 1


def initialize_vllm_model(
    model_name: str, gpu_memory_utilization: float, tensor_parallel_size: int
):
    """Initialize the vLLM model with specified parameters."""
    try:
        from vllm import LLM

        model = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=False,
        )
        logger.info(f"Successfully initialized vLLM model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize vLLM model: {str(e)}")
        raise


def get_critiques(
    model, story: str, language: str, temperature: float, min_p: float
) -> List[str]:
    """
    Query the LLM to get specific critiques of the story.
    Returns a list of 5 critique points ordered by significance.
    """

    system_prompt = (
        "You are a critical literary analyst who specializes in identifying weaknesses in fiction. "
        "Your job is to identify specific problems in the text, not praise its strengths."
    )

    user_prompt = (
        f"Read this story carefully (in {language}) and identify the 5 most significant weaknesses or issues. "
        f"List them in order of significance, with the most serious issue first. "
        f"Consider problems in these categories:\n"
        f"- Language issues:\n"
        f"  * Clichés and overused phrases\n"
        f"  * Purple prose (overly ornate or flowery language)\n"
        f"  * Awkward phrasing or sentence structure\n"
        f"  * Repetitive words or patterns\n"
        f"  * Grammar or spelling errors\n"
        f"  * Inconsistent tone or voice\n"
        f"- Character problems:\n"
        f"  * Flat or one-dimensional characters\n"
        f"  * Inconsistent character behavior\n"
        f"  * Unrealistic motivations or reactions\n"
        f"  * Poor character development\n"
        f"  * Stereotypical or stock characters\n"
        f"- Plot weaknesses:\n"
        f"  * Contrived coincidences\n"
        f"  * Plot holes or logical inconsistencies\n"
        f"  * Rushed or unsatisfying endings\n"
        f"  * Predictable storylines\n"
        f"  * Slow pacing or sections that drag\n"
        f"  * Lack of conflict or tension\n"
        f"- Setting problems:\n"
        f"  * Generic or vague world-building\n"
        f"  * Historically inaccurate details\n"
        f"  * Inconsistent rules or physics\n"
        f"  * Insufficient description or sense of place\n"
        f"- Dialogue issues:\n"
        f"  * Unnatural speech patterns\n"
        f"  * Characters who all sound the same\n"
        f"  * On-the-nose dialogue (too direct/expository)\n"
        f"  * Dialogue that doesn't advance plot or reveal character\n"
        f"- Structural problems:\n"
        f"  * Poor transitions between scenes\n"
        f"  * Uneven focus or balance\n"
        f"  * Unnecessary scenes or tangents\n"
        f"  * Telling instead of showing\n\n"
        f"Format your response as a numbered list with exactly 5 points. For each weakness, provide a specific example "
        f"from the text in quotes. Be concise but precise.\n\n"
        f"Do NOT include any introduction, conclusion, or additional text - ONLY the 5 numbered points ordered by significance."
        f"\n\nStory to analyze:\n\n{story}"
    )

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        min_p=min_p,
        max_tokens=1024,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # First attempt with system prompt
            chat_format = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Log the prompt being sent to the model
            logger.debug(
                f"CRITIQUE PROMPT: System: {system_prompt}\nUser: {user_prompt[:1000]}..."
            )

            try:
                response = (
                    model.chat(
                        chat_format, sampling_params=sampling_params, use_tqdm=False
                    )[0]
                    .outputs[0]
                    .text.strip()
                )
            except Exception as e:
                error_str = str(e)
                logger.warning(f"Error with system prompt: {error_str}")

                # If error mentions system role, try without it
                if "system role" in error_str.lower():
                    logger.info(
                        "Model doesn't support system prompts. Using user prompt only."
                    )

                    # Create a combined prompt for the user message
                    combined_prompt = f"{system_prompt}\n\n{user_prompt}"

                    # Try again with only user prompt
                    response = (
                        model.chat(
                            [{"role": "user", "content": combined_prompt}],
                            sampling_params=sampling_params,
                            use_tqdm=False,
                        )[0]
                        .outputs[0]
                        .text.strip()
                    )
                else:
                    # If it's another kind of error, re-raise it
                    raise

            # Log the full model response
            logger.debug(f"CRITIQUE RESPONSE: {response}")

            # Parse the numbered list from the response
            weaknesses = []
            current_weakness = ""
            for line in response.split("\n"):
                line = line.strip()
                # Check if line starts with a number followed by a period or colon
                if re.match(r"^[1-5][\.:\)]", line):
                    # If we were building a previous weakness, add it to the list
                    if current_weakness:
                        weaknesses.append(current_weakness.strip())
                    # Start a new weakness, removing the number prefix
                    current_weakness = re.sub(r"^[1-5][\.:\)]\s*", "", line)
                else:
                    # Continue building the current weakness
                    current_weakness += " " + line

            # Add the last weakness if there is one
            if current_weakness:
                weaknesses.append(current_weakness.strip())

            # Ensure we have exactly 5 weaknesses
            if len(weaknesses) < 5:
                logger.warning(f"Only found {len(weaknesses)} weaknesses, expected 5")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                # Pad with empty strings if needed
                weaknesses.extend([""] * (5 - len(weaknesses)))
            elif len(weaknesses) > 5:
                logger.warning(
                    f"Found {len(weaknesses)} weaknesses, expected 5. Truncating."
                )
                weaknesses = weaknesses[:5]

            return weaknesses

        except Exception as e:
            logger.error(
                f"Error getting critiques (attempt {attempt+1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retrying
            else:
                logger.error("Max retries exceeded. Returning empty critiques.")
                return [""] * 5


def query_llm_for_ratings(
    model,
    story: str,
    language: str,
    weaknesses: List[str],
    temperature: float,
    min_p: float,
) -> List[int]:
    """Query the LLM with story and return 1-5 ratings for evaluation criteria, informed by weaknesses."""
    # Select criteria (include translation question for non-English)
    criteria = EVALUATION_CRITERIA.copy()
    if language != "en":
        criteria.append(TRANSLATION_QUESTION)

    # Format the weaknesses as a numbered list
    weaknesses_text = "\n".join(
        [f"{i+1}. {weakness}" for i, weakness in enumerate(weaknesses) if weakness]
    )

    # Create the evaluation prompt with clear instructions for structured output
    system_prompt = (
        "You are an objective story evaluator. Based on the provided critique, "
        "rate each criterion on a scale of 1-5, where "
        "1 is the lowest score and 5 is the highest score. Your response must follow the exact format:\n\n"
        "1: [rating 1-5]\n"
        "2: [rating 1-5]\n"
        "...\n\n"
        "Provide ONLY the criterion number and numerical rating for each. "
        "Do not include any explanations or additional text."
    )

    user_prompt = (
        f"A literary critique identified these key weaknesses in the text, ordered by significance (most significant first):\n\n{weaknesses_text}\n\n"
        f"Based on this critical analysis, evaluate this text by rating each criterion on a scale of 1-5, "
        f"where 1 is the lowest score and 5 is the highest score. Be honest in your assessment "
        f"and use the full range of the scale.\n\n"
        f"Text (in {language}):\n{story}\n\n"
        f"Criteria to rate on a 1-5 scale:\n"
    )

    # Add numbered criteria to the prompt with examples
    for i, criterion in enumerate(criteria, 1):
        user_prompt += f"{i}: {criterion['question']}\n"
        user_prompt += f"   1 = {criterion['examples']['1']}\n"
        user_prompt += f"   3 = {criterion['examples']['3']}\n"
        user_prompt += f"   5 = {criterion['examples']['5']}\n\n"

    user_prompt += (
        "\nIMPORTANT: Format your answer as a numbered list like this:\n"
        "1: 4\n"
        "2: 3\n"
        "3: 5\n"
        "...\n"
        "Use ONLY numerical ratings 1-5 for each criterion. Do not include any other text."
    )

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        min_p=min_p,
        max_tokens=1024,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # First attempt with system prompt
            chat_format = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Log the prompt being sent to the model
            logger.debug(
                f"RATING PROMPT: System: {system_prompt}\nUser prompt first 1000 chars: {user_prompt[:1000]}..."
            )

            try:
                response = (
                    model.chat(
                        chat_format, sampling_params=sampling_params, use_tqdm=False
                    )[0]
                    .outputs[0]
                    .text.strip()
                )
            except Exception as e:
                error_str = str(e)
                logger.warning(f"Error with system prompt: {error_str}")

                # If error mentions system role, try without it
                if "system role" in error_str.lower():
                    logger.info(
                        "Model doesn't support system prompts. Using user prompt only."
                    )

                    # Create a combined prompt for the user message
                    combined_prompt = f"{system_prompt}\n\n{user_prompt}"

                    # Try again with only user prompt
                    response = (
                        model.chat(
                            [{"role": "user", "content": combined_prompt}],
                            sampling_params=sampling_params,
                            use_tqdm=False,
                        )[0]
                        .outputs[0]
                        .text.strip()
                    )
                else:
                    # If it's another kind of error, re-raise it
                    raise

            # Log the full model response
            logger.debug(f"RATING RESPONSE: {response}")

            # Initialize ratings with zeros
            ratings = [0] * len(criteria)

            # Clear parsing method: Look for "n: [1-5]" pattern
            pattern = r"(\d+)\s*:\s*([1-5])"
            matches = re.findall(pattern, response.lower())

            for match in matches:
                try:
                    criterion_num = int(match[0]) - 1  # Convert to 0-based index
                    rating = int(match[1])
                    if 0 <= criterion_num < len(criteria):
                        ratings[criterion_num] = rating
                except Exception as e:
                    logger.error(f"Error parsing match {match}: {e}")

            # Log the parsed ratings
            logger.debug(f"Parsed ratings: {ratings}")

            # Log the ratings and check if we need to retry
            if all(r == 0 for r in ratings):
                logger.warning(f"Failed to parse ratings from response: {response}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue

            return ratings

        except Exception as e:
            logger.error(f"Error querying LLM (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retrying
            else:
                logger.error("Max retries exceeded. Returning zeros.")
                return [0] * len(criteria)


def save_current_progress(df, output_file, idx):
    """Save the current progress to the output file."""
    try:
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
        logger.info(f"Progress saved after processing row {idx+1}")
    except Exception as e:
        logger.error(f"Error saving progress after row {idx+1}: {e}")


def evaluate_stories(
    input_file: str,
    output_file: str,
    model_name: str,
    temperature: float,
    min_p: float,
    gpu_memory_utilization: float,
    num_gpus: int,
    save_frequency: int = 1,
):
    """Evaluate stories from input CSV and save results with evaluation columns."""
    try:
        # Initialize model
        model = initialize_vllm_model(
            model_name=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=num_gpus,
        )

        # Read input CSV
        df = pd.read_csv(input_file, quotechar='"', escapechar="\\")
        logger.info(f"Successfully read input file: {input_file} with {len(df)} rows")

        # Create evaluation criterion column names
        criterion_columns = [f"q{i+1}" for i in range(len(EVALUATION_CRITERIA))]
        weakness_columns = [f"w{i+1}" for i in range(5)]  # 5 weaknesses

        # Add extra criterion for non-English texts
        translation_column = "q_translation"

        # Add columns for weaknesses, ratings, and overall score
        for col in weakness_columns:
            df[col] = ""
        for col in criterion_columns:
            df[col] = 0
        df[translation_column] = 0  # Will only be used for non-English texts
        df["length_score"] = 0
        df["overall_score"] = 0.0

        # Process each row individually
        for idx in tqdm(range(len(df)), desc="Evaluating stories"):
            try:
                row = df.iloc[idx]
                story_text = row["story_text"]
                target_word_count = row["target_word_count"]
                language = row["language"]

                # Calculate length score (1-5 scale)
                length_score = calculate_length_score(story_text, target_word_count)
                df.at[idx, "length_score"] = length_score

                # Get critiques for the story
                logger.info(f"Getting critiques for story {idx+1}/{len(df)}")
                weaknesses = get_critiques(
                    model, story_text, language, temperature, min_p
                )

                # Add weaknesses to dataframe
                for i, weakness in enumerate(weaknesses):
                    df.at[idx, weakness_columns[i]] = weakness

                # Evaluate story using the weaknesses to inform ratings
                logger.info(f"Getting ratings for story {idx+1}/{len(df)}")
                ratings = query_llm_for_ratings(
                    model, story_text, language, weaknesses, temperature, min_p
                )

                # Add ratings to dataframe
                for i, rating in enumerate(ratings):
                    if i < len(criterion_columns):
                        df.at[idx, criterion_columns[i]] = rating
                    elif language != "en":  # If this is the translation question
                        df.at[idx, translation_column] = rating

                # Calculate overall score including length score
                all_ratings = ratings + [length_score]
                overall_score = np.mean(all_ratings)
                df.at[idx, "overall_score"] = overall_score

                logger.info(
                    f"Processed row {idx+1}/{len(df)}: Overall Score: {overall_score:.3f}"
                )

                # Save progress after processing each row or according to save_frequency
                if (idx + 1) % save_frequency == 0 or idx == len(df) - 1:
                    save_current_progress(df, output_file, idx)

            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                # Save progress even if there's an error
                save_current_progress(df, output_file, idx)
                continue

        # Final save is redundant now but kept for safety
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
        logger.info(f"Successfully saved all evaluation results to: {output_file}")

    except Exception as e:
        logger.error(f"Error in evaluate_stories: {e}")
        raise


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate stories for AI generated content"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input CSV file with stories"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file for evaluation results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature for generation (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=DEFAULT_MIN_P,
        help=f"Min-p value for sampling (default: {DEFAULT_MIN_P})",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help=f"GPU memory utilization (default: {DEFAULT_GPU_MEMORY_UTILIZATION})",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=DEFAULT_NUM_GPUS,
        help=f"Number of GPUs to use (default: {DEFAULT_NUM_GPUS})",
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=1,
        help="Save CSV after every N rows (default: 1)",
    )
    args = parser.parse_args()

    # Log the configuration
    logger.info("Starting evaluation with configuration:")
    logger.info(f"  Input file: {args.input}")
    logger.info(f"  Output file: {args.output}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Min-p: {args.min_p}")
    logger.info(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    logger.info(f"  Number of GPUs: {args.num_gpus}")
    logger.info(f"  Save frequency: {args.save_frequency}")

    # Evaluate stories
    evaluate_stories(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        temperature=args.temperature,
        min_p=args.min_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_gpus=args.num_gpus,
        save_frequency=args.save_frequency,
    )


if __name__ == "__main__":
    main()
