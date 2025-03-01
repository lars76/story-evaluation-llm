import random
import csv
import logging
import argparse
import gc
import torch
import time
import hashlib
from typing import List
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("story_generation.log"), logging.StreamHandler()],
)
logger = logging.getLogger("story_generator")

# Constants
LANGUAGES = ["en", "de", "es"]

TIME_PERIODS = {
    "en": [
        "medieval times",
        "the renaissance",
        "the industrial revolution",
        "the 1920s",
        "the present day",
        "the near future",
        "the distant future",
        "ancient Rome",
        "Victorian England",
    ],
    "de": [
        "dem Mittelalter",
        "der Renaissance",
        "der industriellen Revolution",
        "den 1920er Jahren",
        "der Gegenwart",
        "der nahen Zukunft",
        "der fernen Zukunft",
        "dem antiken Rom",
        "dem viktorianischen Zeitalter",
    ],
    "es": [
        "la época medieval",
        "el renacimiento",
        "la revolución industrial",
        "los años 20",
        "la actualidad",
        "el futuro cercano",
        "el futuro lejano",
        "la antigua Roma",
        "la época victoriana",
    ],
}

THEMES = {
    "en": [
        "adventure",
        "horror",
        "thriller",
        "magic",
        "romance",
        "crime",
        "science fiction",
        "fantasy",
        "mystery",
        "historical fiction",
        "comedy",
        "drama",
        "supernatural",
        "dystopian",
        "western",
    ],
    "de": [
        "Abenteuer",
        "Horror",
        "Thriller",
        "Magie",
        "Romantik",
        "Krimi",
        "Science-Fiction",
        "Fantasy",
        "Mystery",
        "historische Fiktion",
        "Komödie",
        "Drama",
        "Übernatürliches",
        "Dystopie",
        "Western",
    ],
    "es": [
        "aventura",
        "terror",
        "suspenso",
        "magia",
        "romance",
        "crimen",
        "ciencia ficción",
        "fantasía",
        "misterio",
        "ficción histórica",
        "comedia",
        "drama",
        "sobrenatural",
        "distopía",
        "western",
    ],
}

# Word count range for stories (min, max)
WORD_RANGE = (1000, 2000)

# Number of story pairs to generate per model by default
DEFAULT_STORIES_PER_MODEL = 50

DEFAULT_MODELS = [
    "AMead10/c4ai-command-r-08-2024-awq",
    "casperhansen/mistral-nemo-instruct-2407-awq",
    "Qwen/Qwen2.5-14B-Instruct-AWQ",
    "Orion-zhen/aya-expanse-32b-AWQ",
    "AMead10/SuperNova-Medius-AWQ",
    "solidrust/Hermes-3-Llama-3.1-8B-AWQ",
    "harborwater/Gemma-2-9B-It-SPPO-Iter3-AWQ",
    "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ",
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "alijawad07/aya-23-8B-AWQ-GEMM",
    "arcee-ai/Arcee-Blitz-AWQ",
    "stelterlab/phi-4-AWQ",
    "solidrust/Gemma-2-Ataraxy-9B-AWQ",
    "Orion-zhen/aya-expanse-8b-AWQ",
    "solidrust/gemma-2-9b-it-AWQ",
]

# Model and generation parameters
MIN_P = 0.05

# Random seeds for generation
SEEDS = [42, 48]  # Expanded set of seeds

# Default temperatures to try
DEFAULT_TEMPERATURES = [0.5, 0.75, 1.0, 1.25]

# Default GPU settings
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_NUM_GPUS = 4

# Prompts for different story generation approaches
PROMPTS = {
    "en": {
        "complete_story": 'Write a complete story of approximately {word_count} words about "{theme}" set in {time_period}. Create a compelling narrative with a clear beginning, middle, and end.',
        "scene_beginning": 'Write the first scene (approximately {word_count} words) of a story about "{theme}" set in {time_period}. Create an engaging beginning that establishes the setting and introduces key characters.',
        "scene_continuation": 'Continue the following story by writing the final scene (approximately {word_count} words). The story is about "{theme}" set in {time_period}.\n\nPrevious scene:\n{previous_scene}\n\nWrite a concluding scene that brings the narrative to a satisfying resolution.',
    },
    "de": {
        "complete_story": 'Schreibe eine vollständige Geschichte von ungefähr {word_count} Wörtern über "{theme}" in {time_period}. Erschaffe eine fesselnde Erzählung mit einem klaren Anfang, Mittelteil und Ende.',
        "scene_beginning": 'Schreibe die erste Szene (ungefähr {word_count} Wörter) einer Geschichte über "{theme}" in {time_period}. Gestalte einen fesselnden Anfang, der das Setting etabliert und die Hauptcharaktere vorstellt.',
        "scene_continuation": 'Setze die folgende Geschichte fort, indem du die letzte Szene (ungefähr {word_count} Wörter) schreibst. Die Geschichte handelt von "{theme}" in {time_period}.\n\nVorherige Szene:\n{previous_scene}\n\nSchreibe eine abschließende Szene, die die Erzählung zu einem zufriedenstellenden Ende führt.',
    },
    "es": {
        "complete_story": 'Escribe una historia completa de aproximadamente {word_count} palabras sobre "{theme}" ambientada en {time_period}. Crea una narrativa cautivadora con un principio, desarrollo y final claros.',
        "scene_beginning": 'Escribe la primera escena (aproximadamente {word_count} palabras) de una historia sobre "{theme}" ambientada en {time_period}. Crea un comienzo atractivo que establezca el escenario y presente a los personajes principales.',
        "scene_continuation": 'Continúa la siguiente historia escribiendo la escena final (aproximadamente {word_count} palabras). La historia trata sobre "{theme}" ambientada en {time_period}.\n\nEscena anterior:\n{previous_scene}\n\nEscribe una escena final que lleve la narrativa a una resolución satisfactoria.',
    },
}


class StoryGenerator:
    def __init__(
        self,
        output_file: str,
        model_name: str,
        temperature: float,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        tensor_parallel_size: int = DEFAULT_NUM_GPUS,
    ):
        self.output_file = output_file
        self.model_name = model_name
        self.temperature = temperature
        self.short_model_name = model_name.split("/")[-1]

        # Initialize model
        try:
            self.model = LLM(
                model=model_name,
                # quantization="awq_marlin",
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,  # Number of GPUs to use
                distributed_executor_backend="mp",
                disable_custom_all_reduce=True,
                enforce_eager=False,
                # max_model_len=8092,
                trust_remote_code=True,
            )
            logger.info(
                f"Successfully initialized vLLM model: {model_name} with temperature {temperature}"
            )
            logger.info(
                f"Using GPU utilization: {gpu_memory_utilization}, GPUs: {tensor_parallel_size}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vLLM model: {str(e)}")
            raise

    def generate_story(self, prompt: str, random_seed: int) -> str:
        """Generate a single story based on the given prompt with specified random seed"""
        # Start with system prompt and user prompt
        system_prompt = [
            {
                "role": "system",
                "content": "You are a creative storyteller who excels at crafting engaging narratives with compelling characters and vivid descriptions. When asked to write a story, provide only the plain text of the story itself without titles, headings, formatting, or commentary.",
            }
        ]
        user_prompt = [{"role": "user", "content": prompt}]
        chat_prompt = system_prompt + user_prompt

        sampling_params = SamplingParams(
            temperature=self.temperature, min_p=MIN_P, max_tokens=4096, seed=random_seed
        )

        try:
            # First attempt with system prompt
            response = (
                self.model.chat(
                    chat_prompt, sampling_params=sampling_params, use_tqdm=False
                )[0]
                .outputs[0]
                .text
            )

            return response.strip()
        except Exception as e:
            error_str = str(e)
            logger.warning(f"Error with system prompt: {error_str}")

            # If error mentions system role, try without it
            if "system role" in error_str.lower():
                logger.info(
                    f"Model {self.short_model_name} doesn't support system prompts. Using user prompt only."
                )

                # Create a combined prompt for the user message
                combined_prompt = (
                    "You are a creative storyteller who excels at crafting engaging narratives with compelling characters and vivid descriptions. When asked to write a story, provide only the plain text of the story itself without titles, headings, formatting, or commentary.\n\n"
                    + prompt
                )

                try:
                    # Second attempt with only user prompt
                    response = (
                        self.model.chat(
                            [{"role": "user", "content": combined_prompt}],
                            sampling_params=sampling_params,
                            use_tqdm=False,
                        )[0]
                        .outputs[0]
                        .text
                    )

                    return response.strip()
                except Exception as e2:
                    logger.error(f"Error generating story with user prompt: {str(e2)}")
                    return f"Error generating story: {str(e2)}"
            else:
                logger.error(f"Error generating story: {error_str}")
                return f"Error generating story: {error_str}"

    def save_story(
        self,
        prompt: str,
        story_text: str,
        target_word_count: int,
        language: str,
        theme: str,
        time_period: str,
        generation_type: str,
        seed: int,
        prompt_id: str = None,
    ):
        """Save a single story to the CSV file"""
        with open(self.output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    prompt_id,
                    prompt,
                    story_text,
                    target_word_count,
                    language,
                    theme,
                    time_period,
                    generation_type,
                    self.model_name,
                    self.temperature,
                    seed,
                ]
            )

    def generate_stories(self, num_stories: int, seeds: List[int]):
        """Generate stories using multiple seeds for each prompt"""
        for i in tqdm(
            range(num_stories),
            desc=f"Generating stories for {self.short_model_name} (temp={self.temperature})",
        ):
            # Randomly select language, theme, and time period
            lang = random.choice(LANGUAGES)
            theme = random.choice(THEMES[lang])
            time_period = random.choice(TIME_PERIODS[lang])

            # Decide whether to generate complete story or scene-based story
            generation_type = random.choice(["complete", "scenes"])

            # Generate target word count within range
            word_count = random.randint(WORD_RANGE[0], WORD_RANGE[1])

            try:
                # Generate the actual prompt text first
                if generation_type == "complete":
                    # For complete stories, generate the full prompt
                    prompt = PROMPTS[lang]["complete_story"].format(
                        word_count=word_count, theme=theme, time_period=time_period
                    )
                    # Create hash directly from the formatted prompt text
                    prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:10]
                    prompt_id = f"p{prompt_hash}"

                    # Generate stories with different random seeds
                    for seed in seeds:
                        story = self.generate_story(prompt, random_seed=seed)
                        self.save_story(
                            prompt,
                            story,
                            word_count,
                            lang,
                            theme,
                            time_period,
                            "complete",
                            seed,
                            prompt_id,
                        )

                else:
                    # Generate first scene
                    scene_word_count = word_count // 2
                    scene1_prompt = PROMPTS[lang]["scene_beginning"].format(
                        word_count=scene_word_count,
                        theme=theme,
                        time_period=time_period,
                    )

                    # Create hash directly from the first scene prompt text
                    scene1_hash = hashlib.md5(
                        scene1_prompt.encode("utf-8")
                    ).hexdigest()[:10]
                    scene1_prompt_id = f"p{scene1_hash}"

                    # Generate first scene with multiple seeds
                    first_scenes = {}
                    for seed in seeds:
                        scene1 = self.generate_story(scene1_prompt, random_seed=seed)
                        self.save_story(
                            scene1_prompt,
                            scene1,
                            scene_word_count,
                            lang,
                            theme,
                            time_period,
                            "scene_beginning",
                            seed,
                            scene1_prompt_id,
                        )
                        first_scenes[seed] = scene1

                    # Generate second scenes using the first scene from the first seed
                    # This maintains consistency in continuations
                    reference_seed = seeds[0]
                    reference_scene = first_scenes[reference_seed]

                    scene2_prompt = PROMPTS[lang]["scene_continuation"].format(
                        word_count=scene_word_count,
                        theme=theme,
                        time_period=time_period,
                        previous_scene=reference_scene,
                    )

                    # Hash the continuation prompt (which includes the first scene content)
                    scene2_hash = hashlib.md5(
                        scene2_prompt.encode("utf-8")
                    ).hexdigest()[:10]
                    scene2_prompt_id = f"p{scene2_hash}"

                    # Generate second scenes with multiple seeds
                    for seed in seeds:
                        scene2 = self.generate_story(scene2_prompt, random_seed=seed)
                        self.save_story(
                            scene2_prompt,
                            scene2,
                            scene_word_count,
                            lang,
                            theme,
                            time_period,
                            "scene_continuation",
                            seed,
                            scene2_prompt_id,
                        )

                logger.info(
                    f"Generated story set {i+1}/{num_stories} in {lang} with {len(seeds)} seeds"
                )
            except Exception as e:
                logger.error(f"Error generating story set {i+1}: {str(e)}")
                continue


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate stories for DPO training")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help="List of model names to use",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=DEFAULT_TEMPERATURES,
        help=f"List of temperatures to use (default: {DEFAULT_TEMPERATURES})",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="all_stories.csv",
        help="Single CSV file to save all results",
    )
    parser.add_argument(
        "--stories_per_model",
        type=int,
        default=DEFAULT_STORIES_PER_MODEL,
        help=f"Number of story prompts to generate per model (default: {DEFAULT_STORIES_PER_MODEL})",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help=f"GPU memory utilization factor (0.0 to 1.0, default: {DEFAULT_GPU_MEMORY_UTILIZATION})",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=DEFAULT_NUM_GPUS,
        help=f"Number of GPUs to use (tensor_parallel_size, default: {DEFAULT_NUM_GPUS})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help=f"Random seeds to use for generation (default: {SEEDS})",
    )
    args = parser.parse_args()

    # Create output file with headers
    output_file = args.output_file
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "prompt_id",
                "prompt",
                "story_text",
                "target_word_count",
                "language",
                "theme",
                "time_period",
                "generation_type",
                "model_name",
                "temperature",
                "seed",
            ]
        )
    logger.info(f"Created output file: {output_file}")

    # Count total combinations for overall progress tracking
    total_combinations = len(args.models) * len(args.temperatures)
    current_combination = 0

    # For each model and temperature combination
    for model_name in args.models:
        # Get the short model name for logging
        short_model_name = model_name.split("/")[-1]

        for temp in args.temperatures:
            current_combination += 1
            logger.info(
                f"Processing combination {current_combination}/{total_combinations}: {short_model_name} at temperature {temp}"
            )

            try:
                # Initialize generator for this model and temperature
                generator = StoryGenerator(
                    output_file=output_file,
                    model_name=model_name,
                    temperature=temp,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    tensor_parallel_size=args.num_gpus,
                )

                # Generate stories
                generator.generate_stories(args.stories_per_model, args.seeds)
                logger.info(
                    f"Story generation complete for {short_model_name} at temperature {temp}"
                )

                # Properly terminate vLLM resources with more complete cleanup
                import contextlib
                from vllm.distributed.parallel_state import (
                    destroy_model_parallel,
                    destroy_distributed_environment,
                )
                import ray

                destroy_model_parallel()
                destroy_distributed_environment()
                del generator.model.llm_engine.model_executor
                del generator.model
                with contextlib.suppress(AssertionError):
                    torch.distributed.destroy_process_group()
                gc.collect()
                torch.cuda.empty_cache()
                ray.shutdown()
                time.sleep(10)

            except Exception as e:
                logger.error(
                    f"Error processing model {short_model_name} at temperature {temp}: {str(e)}"
                )
                continue

    logger.info(f"All story generation complete. All stories saved to {output_file}")


if __name__ == "__main__":
    main()
