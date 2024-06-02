import hashlib
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


def identify_speaker(
    enrollment_dir: str,
    test_dir: str,
    save_dir: Optional[str] = None,
    device: Optional[str] = None,
    hf_api_token: Optional[str] = None,
    show_tqdm: bool = True
) -> List[Tuple[str, float]]:
    """
    Compute embeddings and distances for audio files for speaker identification.

    Parameters:
        enrollment_dir (str): Path to the directory containing the enrollment audio files.
        test_dir (str): Path to the directory containing the test audio files.
        save_dir (str, optional): Path to the directory where embeddings and distances will be saved.
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
    os.makedirs(enrollment_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

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

    if save_dir:
        # Load cached embeddings
        cached_embeddings_path = os.path.join(save_dir, "embeddings_cache.pkl")
        cached_embeddings: Dict[bytes, np.ndarray] = dict()
        if os.path.exists(cached_embeddings_path):
            with open(cached_embeddings_path, "rb") as f:
                cached_embeddings = pickle.load(f)
            logger.info(f"Loaded {len(cached_embeddings)} cached embeddings from {cached_embeddings_path}")

    # audio_index = {audio_path: (audio_embedding, audio_hash)}
    enrollment_index: Dict[str, Tuple[Optional[np.ndarray], bytes]] = dict()
    test_index: Dict[str, Tuple[Optional[np.ndarray], bytes]] = dict()
    pairs = [(enrollment_dir, enrollment_index), (test_dir, test_index)]

    # Compute embeddings for enrollment and test audio files
    for audio_dir, audio_index in pairs:
        audio_type = "enrollment" if audio_dir == enrollment_dir else "test"

        # Compute audio hashes
        logger.info(f"Computing {audio_type} audio hashes")
        audio_hashes = compute_hashes(audio_dir, show_tqdm)
        audio_index.update({
            audio_path: (None, audio_hash)
            for audio_path, audio_hash in audio_hashes.items()
        })

        # Check for cached embeddings
        if save_dir:
            for audio_path, audio_data in audio_index.items():
                _, audio_hash = audio_data
                if audio_hash in cached_embeddings:
                    audio_embedding = cached_embeddings[audio_hash]
                    audio_index[audio_path] = (audio_embedding, audio_hash)

        # Compute embeddings
        logger.info(f"Computing {audio_type} embeddings")
        compute_embeddings(audio_index, inference, show_tqdm)

        # Serialize embeddings
        if save_dir:
            cached_embeddings.update({
                audio_hash: audio_embedding
                for _, (audio_embedding, audio_hash) in audio_index.items()
            })
            with open(cached_embeddings_path, "wb") as f:
                pickle.dump(cached_embeddings, f)
            logger.info(f"Cached {audio_type} embeddings saved to {cached_embeddings_path}")

    # Compute distances
    logger.info("Computing distances")
    distances = compute_distances(enrollment_index, test_index, show_tqdm)
    sorted_distances = sorted(
        distances.items(),
        key=lambda x: float("inf") if np.isnan(x[1]) else x[1]
    )

    # Serialize distances
    if save_dir:
        distances_path = os.path.join(save_dir, "distances.json")
        with open(distances_path, "w") as f:
            json.dump(sorted_distances, f, indent=4)
        logger.info(f"Distances saved to {distances_path}")

    return sorted_distances


def compute_hashes(
    audio_dir: str,
    show_tqdm: bool = True,
    tqdm_kwargs: Dict[str, Any] = None
) -> Dict[str, bytes]:
    """
    Compute audio hashes for audio files in a given directory.

    Parameters:
        audio_dir (str): Path to the directory containing the audio files.
        show_tqdm (bool, optional): Whether to show a tqdm progress bar.
        tqdm_kwargs (Dict[str, Any], optional): Additional arguments for the tqdm progress bar.

    Returns:
        Dict[str, bytes]: Dictionary of audio hashes. Keys are audio file paths, values are audio hashes.
    """
    iterator = tqdm(os.listdir(audio_dir), **(tqdm_kwargs or {})) \
        if show_tqdm else os.listdir(audio_dir)

    hashes: Dict[str, bytes] = dict()
    extensions = (".flac", ".mp3", ".ogg", ".wav")
    for file_name in iterator:
        if not file_name.endswith(extensions):
            continue
        audio_path = os.path.join(audio_dir, file_name)
        with open(audio_path, "rb") as f:
            audio_hash = hashlib.sha256(f.read()).digest()
        hashes[audio_path] = audio_hash
    return hashes


def compute_embeddings(
    audio_index: Dict[str, Tuple[Optional[np.ndarray], bytes]],
    inference: Inference,
    show_tqdm: bool = True,
    tqdm_kwargs: Dict[str, Any] = None
) -> None:
    """
    Compute embeddings for audio files in a given directory.

    Parameters:
        audio_index (Dict[str, Tuple[Optional[np.ndarray], bytes]]): Dictionary to modify in-place with computed
            embeddings. Keys are audio file paths, values are tuples of audio embeddings and audio file hashes.
        inference (Inference): Inference object used to compute embeddings.
        show_tqdm (bool, optional): Whether to show a tqdm progress bar.
        tqdm_kwargs (Dict[str, Any], optional): Additional arguments for the tqdm progress bar.

    Returns:
        None: The embeddings are modified in-place.
    """
    iterator = tqdm(audio_index.items(), **(tqdm_kwargs or {})) \
        if show_tqdm else audio_index.items()

    for audio_path, audio_data in iterator:
        audio_embedding, audio_hash = audio_data
        if audio_embedding is None:
            try:
                audio_embedding = inference(audio_path)
            except Exception as e:
                error_message = f"Failed to compute embeddings for {audio_path}: {e}"
                tqdm.write(error_message)
                original_level = logger.level
                logger.setLevel(logging.CRITICAL)
                logger.error(error_message)
                logger.setLevel(original_level)
                continue
        audio_index[audio_path] = (audio_embedding, audio_hash)


def compute_distances(
    enrollment_index: Dict[str, Tuple[Optional[np.ndarray], bytes]],
    test_index: Dict[str, Tuple[Optional[np.ndarray], bytes]],
    show_tqdm: bool = True,
    tqdm_kwargs: Dict[str, Any] = None
) -> Dict[str, float]:
    """
    Compute average cosine distances between enrollment and test embeddings.

    Parameters:
        enrollment_index (Dict[str, Tuple[Optional[np.ndarray], bytes]]): Dictionary where keys are paths to enrollment
            audio files and values are tuples of either None or precomputed embeddings and audio file hashes.
        test_index (Dict[str, Tuple[Optional[np.ndarray], bytes]]): Dictionary where keys are paths to test audio files
            and values are tuples of either None or precomputed embeddings and audio file hashes.
        show_tqdm (bool, optional): Whether to show a tqdm progress bar.
        tqdm_kwargs (Dict[str, Any], optional): Additional arguments for the tqdm progress bar.

    Returns:
        Dict[str, float]: Dictionary of distances. Keys are test file paths, values are the average cosine distances
            from the test embeddings to the enrollment embeddings. Lower distances indicate better matches.
    """
    iterator = tqdm(enrollment_index.items(), **(tqdm_kwargs or {})) \
        if show_tqdm else enrollment_index.items()

    distances = {}
    for enrollment_path, enrollment_data in iterator:
        enrollment_embedding, _ = enrollment_data
        for test_path, test_data in test_index.items():
            test_embedding, _ = test_data
            distance = cosine(enrollment_embedding.flatten(), test_embedding.flatten())
            distances[test_path] = (distances.get(test_path) or 0) + distance
    distances = {k: v / len(enrollment_index) for k, v in distances.items()}
    return distances


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute embeddings and distances for speech audio files for speaker identification."
    )
    parser.add_argument(
        "--enrollment-dir",
        type=str,
        required=True,
        help="Path to the directory containing the enrollment audio files."
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Path to the directory containing the test audio files."
    )
    parser.add_argument(
        "--save-dir",
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
        enrollment_dir=args.enrollment_dir,
        test_dir=args.test_dir,
        save_dir=args.save_dir,
        device=args.device,
        hf_api_token=args.hf_api_token
    )
