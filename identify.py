import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from pyannote.audio import Model, Inference
from scipy.spatial.distance import cosine
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def compute_embeddings(
    directory: str,
    embeddings: Dict[str, np.ndarray],
    inference: Inference,
    show_tqdm: bool = True,
    tqdm_kwargs: Dict[str, Any] = None
) -> None:
    """
    Compute embeddings for audio files in a given directory.

    Parameters:
        directory (str): Path to the directory containing the audio files.
        embeddings (Dict[str, np.ndarray]): Dictionary to modify in-place with computed embeddings.
            Keys are file paths, values are embeddings.
        inference (Inference): Inference object used to compute embeddings.
        show_tqdm (bool, optional): Whether to show a tqdm progress bar.
        tqdm_kwargs (Dict[str, Any], optional): Additional arguments for the tqdm progress bar.

    Returns:
        None: The embeddings are modified in-place.
    """
    if show_tqdm:
        iterator = tqdm(os.listdir(directory), **(tqdm_kwargs or {}))
    else:
        iterator = os.listdir(directory)

    extensions = (".flac", ".mp3", ".ogg", ".wav")
    for file_name in iterator:
        if not file_name.endswith(extensions):
            continue
        file_path = os.path.join(directory, file_name)

        embedding = embeddings.get(file_path)
        if embedding is None:
            try:
                embedding = inference(file_path)
            except Exception as e:
                message = f"Failed to compute embeddings for {file_path}: {e}"
                tqdm.write(message)
                original_level = logger.level
                logger.setLevel(logging.CRITICAL)
                logger.error(message)
                logger.setLevel(original_level)
                continue
        embeddings[file_path] = embedding


def compute_distances(
    enrollment_embeddings: Dict[str, np.ndarray],
    test_embeddings: Dict[str, np.ndarray],
    show_tqdm: bool = True,
    tqdm_kwargs: Dict[str, Any] = None
) -> Dict[str, float]:
    """
    Compute average cosine distances between enrollment and test embeddings.

    Parameters:
        enrollment_embeddings (Dict[str, np.ndarray]): Dictionary of enrollment embeddings.
            Keys are file paths, values are enrollment embeddings.
        test_embeddings (Dict[str, np.ndarray]): Dictionary of test embeddings.
            Keys are file paths, values are test embeddings.
        show_tqdm (bool, optional): Whether to show a tqdm progress bar.
        tqdm_kwargs (Dict[str, Any], optional): Additional arguments for the tqdm progress bar.

    Returns:
        Dict[str, float]: Dictionary of distances. Keys are test file paths, values are the average cosine distances
            from the test embeddings to the enrollment embeddings. Lower distances indicate better matches.
    """
    if show_tqdm:
        iterator = tqdm(enrollment_embeddings.items(), **(tqdm_kwargs or {}))
    else:
        iterator = enrollment_embeddings.items()

    distances = {}
    for enrollment_path, enrollment_embedding in iterator:
        for test_path, test_embedding in test_embeddings.items():
            distance = cosine(enrollment_embedding.flatten(), test_embedding.flatten())
            if np.isnan(distance):
                distance = float("inf")
            distances[test_path] = (distances.get(test_path) or 0) + distance
    distances = {k: v / len(enrollment_embeddings) for k, v in distances.items()}
    return distances


def identify_speaker(
    enrollment_path: str,
    test_path: str,
    save_path: Optional[str] = None,
    device: Optional[str] = None,
    hf_api_token: Optional[str] = None,
    show_tqdm: bool = True
) -> List[Tuple[str, float]]:
    """
    Compute embeddings and distances for audio files for speaker identification.

    Parameters:
        enrollment_path (str): Path to the directory containing the enrollment audio files.
        test_path (str): Path to the directory containing the test audio files.
        save_path (str, optional): Path to the directory where embeddings and distances will be saved.
        device (str, optional): Device to use for inference. Defaults to "cuda" if available, else "cpu".
        hf_api_token (str, optional): Hugging Face API token for model authentication.
        show_tqdm (bool, optional): Whether to show a tqdm progress bar.

    Returns:
        List[Tuple[str, float]]: List of test file paths with corresponding distances. Lower is better.
    """
    load_dotenv()

    # Defaults
    hf_api_token = hf_api_token or os.environ.get("HF_API_TOKEN")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Create directories
    os.makedirs(enrollment_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    # Load model
    model = Model.from_pretrained(
        checkpoint="pyannote/embedding",
        use_auth_token=hf_api_token,
    )
    inference = Inference(
        model,
        window="whole",
        device=device
    )
    logger.info(f"Loaded model to {device}")

    enrollment_embeddings = dict()
    test_embeddings = dict()

    # Load embeddings if they exist
    if save_path is not None:
        enrollment_embeddings_path = os.path.join(save_path, "enrollment_embeddings.pkl")
        test_embeddings_path = os.path.join(save_path, "test_embeddings.pkl")

        if os.path.exists(enrollment_embeddings_path):
            with open(enrollment_embeddings_path, "rb") as f:
                enrollment_embeddings = pickle.load(f)
            logger.info(f"Loaded {len(enrollment_embeddings)} enrollment embeddings from {enrollment_embeddings_path}")
        if os.path.exists(test_embeddings_path):
            with open(test_embeddings_path, "rb") as f:
                test_embeddings = pickle.load(f)
            logger.info(f"Loaded {len(test_embeddings)} test embeddings from {test_embeddings_path}")

    # Compute enrollment embeddings
    compute_embeddings(
        enrollment_path,
        enrollment_embeddings,
        inference,
        show_tqdm=show_tqdm,
        tqdm_kwargs={
            "desc": "Computing enrollment embeddings"
        }
    )

    # Serialize enrollment embeddings
    if save_path is not None:
        with open(enrollment_embeddings_path, "wb") as f:
            pickle.dump(enrollment_embeddings, f)
        logger.info(f"Enrollment embeddings saved to {enrollment_embeddings_path}")

    # Compute test embeddings
    compute_embeddings(
        test_path,
        test_embeddings,
        inference,
        show_tqdm=show_tqdm,
        tqdm_kwargs={
            "desc": "Computing test embeddings"
        }
    )

    # Serialize test embeddings
    if save_path is not None:
        with open(test_embeddings_path, "wb") as f:
            pickle.dump(test_embeddings, f)
        logger.info(f"Test embeddings saved to {test_embeddings_path}")

    # Compute distances
    distances = compute_distances(
        enrollment_embeddings,
        test_embeddings,
        show_tqdm=show_tqdm,
        tqdm_kwargs={
            "desc": "Computing distances"
        }
    )
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])

    # Serialize distances
    if save_path is not None:
        for i, (path, distance) in enumerate(sorted_distances):
            if distance == float("inf"):
                sorted_distances[i] = (path, None)
        distances_path = os.path.join(save_path, "distances.json")
        with open(distances_path, "w") as f:
            json.dump(sorted_distances, f, indent=4)
        logger.info(f"Distances saved to {distances_path}")

    return sorted_distances


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute embeddings and distances for speech audio files for speaker identification."
    )
    parser.add_argument(
        "--enrollment-path",
        type=str,
        required=True,
        help="Path to the directory containing the enrollment audio files."
    )
    parser.add_argument(
        "--test-path",
        type=str,
        required=True,
        help="Path to the directory containing the test audio files."
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to the directory where embeddings and distances will be saved."
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Device to use for inference."
    )
    parser.add_argument(
        "--hf-api-token",
        type=str,
        required=False,
        help="Hugging Face API token for model authentication."
    )
    args = parser.parse_args()

    identify_speaker(
        enrollment_path=args.enrollment_path,
        test_path=args.test_path,
        save_path=args.save_path,
        device=args.device,
        hf_api_token=args.hf_api_token
    )
