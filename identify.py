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


def hash_file(
    file_path: str
) -> bytes:
    """
    Compute the SHA-256 hash of a file.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        bytes: SHA-256 hash of the file.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.digest()


def compute_embeddings(
    audio_index: Dict[str, Tuple[np.ndarray, bytes]],
    inference: Inference,
    show_tqdm: bool = True,
    tqdm_kwargs: Dict[str, Any] = None
) -> None:
    """
    Compute embeddings for audio files in a given directory.

    Parameters:
        audio_index (Dict[str, Tuple[np.ndarray, bytes]]): Dictionary to modify in-place with computed embeddings.
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
    enrollment_index: Dict[str, Tuple[np.ndarray, bytes]],
    test_index: Dict[str, Tuple[np.ndarray, bytes]],
    show_tqdm: bool = True,
    tqdm_kwargs: Dict[str, Any] = None
) -> Dict[str, float]:
    """
    Compute average cosine distances between enrollment and test embeddings.

    Parameters:
        enrollment_index (Dict[str, Tuple[np.ndarray, bytes]]): Dictionary of enrollment embeddings and file hashes.
        test_index (Dict[str, Tuple[np.ndarray, bytes]]): Dictionary of test embeddings and file hashes.
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
            if np.isnan(distance):
                distance = float("inf")
            distances[test_path] = (distances.get(test_path) or 0) + distance
    distances = {k: v / len(enrollment_index) for k, v in distances.items()}
    return distances


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
    if save_dir is not None:
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

    # file_path: (embedding, file_hash)
    enrollment_index: Dict[str, Tuple[Optional[np.ndarray], bytes]] = dict()
    test_index: Dict[str, Tuple[Optional[np.ndarray], bytes]] = dict()

    # Compute hashes
    pairs = [
        (enrollment_dir, enrollment_index),
        (test_dir, test_index)
    ]
    extensions = (".flac", ".mp3", ".ogg", ".wav")
    for audio_dir, audio_index in pairs:
        for file_name in os.listdir(audio_dir):
            if not file_name.endswith(extensions):
                continue
            audio_path = os.path.join(audio_dir, file_name)
            audio_hash = hash_file(audio_path)
            audio_index[audio_path] = (None, audio_hash)

    # Load embeddings if they exist
    cached_embeddings_path = os.path.join(save_dir, "embeddings_cache.pkl") if save_dir else None
    cached_embeddings: Dict[bytes, np.ndarray] = dict()
    if save_dir and os.path.exists(cached_embeddings_path):
        with open(cached_embeddings_path, "rb") as f:
            cached_embeddings = pickle.load(f)
        logger.info(f"Loaded {len(cached_embeddings)} cached embeddings from {cached_embeddings_path}")

        for audio_dir, audio_index in pairs:
            for audio_path, audio_data in audio_index.items():
                _, audio_hash = audio_data
                audio_embedding = cached_embeddings.get(audio_hash)
                if audio_embedding is not None:
                    audio_index[audio_path] = (audio_embedding, audio_hash)

    # Compute enrollment embeddings
    compute_embeddings(
        enrollment_index,
        inference,
        show_tqdm=show_tqdm,
        tqdm_kwargs={
            "desc": "Computing enrollment embeddings"
        }
    )

    # Serialize enrollment embeddings
    if save_dir is not None:
        cached_embeddings.update({
            audio_hash: audio_embedding for _, (audio_embedding, audio_hash) in enrollment_index.items()
        })
        with open(cached_embeddings_path, "wb") as f:
            pickle.dump(cached_embeddings, f)
        logger.info(f"Cached enrollment embeddings saved to {cached_embeddings_path}")

    # Compute test embeddings
    compute_embeddings(
        test_index,
        inference,
        show_tqdm=show_tqdm,
        tqdm_kwargs={
            "desc": "Computing test embeddings"
        }
    )

    # Serialize test embeddings
    if save_dir is not None:
        cached_embeddings.update({
            audio_hash: audio_embedding for _, (audio_embedding, audio_hash) in test_index.items()
        })
        with open(cached_embeddings_path, "wb") as f:
            pickle.dump(cached_embeddings, f)
        logger.info(f"Cached test embeddings saved to {cached_embeddings_path}")

    # Compute distances
    distances = compute_distances(
        enrollment_index,
        test_index,
        show_tqdm=show_tqdm,
        tqdm_kwargs={
            "desc": "Computing distances"
        }
    )
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])

    # Serialize distances
    if save_dir is not None:
        for i, (audio_dir, distance) in enumerate(sorted_distances):
            if distance == float("inf"):
                sorted_distances[i] = (audio_dir, None)
        distances_path = os.path.join(save_dir, "distances.json")
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
        enrollment_dir=args.enrollment_path,
        test_dir=args.test_path,
        save_dir=args.save_path,
        device=args.device,
        hf_api_token=args.hf_api_token
    )
