# speaker-identification

This repository contains a simple script that performs one-to-many speaker matching on audio files. The process involves comparing the speech samples from a singular individual, referred to as the "enrollment audio", against a collection of speech samples from a group of individuals, referred to as the "test audio", in order to determine which of the test audio belong to the same speaker as the enrollment audio.

The script performs the following steps:

1. Generate embeddings for all speech samples using the [`pyannote/embedding`](https://huggingface.co/pyannote/embedding) model from Hugging Face.
2. Compute the average cosine distance between each test audio and the enrollment audio using the embeddings.
3. Rank the test audio files in ascending order based on the distances. Smaller values indicate a closer match to the enrollment audio.
4. If an output path is provided, serialize the sorted list of test audio files with their corresponding distances as a JSON file, and save the embeddings as a pickle file.

## Installation

Before installing the required dependencies, install PyTorch with a GPU compute platform from their [website](https://pytorch.org/get-started/locally) for improved performance. The remaining packages can be installed using the following command:

```bash
pip install -r requirements.txt
```

Additionally, the embedding model requires acceptance of its conditions found in the [model card](https://huggingface.co/pyannote/embedding). As such, a Hugging Face API token is needed due to the model's access level. Obtain the token from the account settings and set it in a `.env` file in the repository root directory as follows:

```env
HF_API_TOKEN=token_goes_here
```

## Usage

The `identify.py` script can be run from the command line as such:

```bash
python identify.py [-h] --enrollment-path ENROLLMENT_PATH --test-path TEST_PATH --save-path SAVE_PATH [--device DEVICE] [--hf-api-token HF_API_TOKEN]
```

- `--enrollment-path`: Path to the directory containing the enrollment audio files.
- `--test-path`: Path to the directory containing the test audio files.
- `--save-path`: Path to the directory where embeddings and distances will be saved.
- `--device`: (Optional) Device to use for inference. Defaults to "cuda" if available, else "cpu".
- `--hf-api-token`: (Optional) Hugging Face API token for model authentication.

Errors may occur during the embedding computation step if the length of the audio file is too short. In such cases, the script will skip over the problematic file and continue with the next one.

## License

This project is licensed under the [MIT License](LICENSE).
