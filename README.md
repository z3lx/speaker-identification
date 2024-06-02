# speaker-identification
This repository contains a simple script that performs speaker identification on audio files. It uses the `pyannote/embedding` model from Hugging Face to calculate embeddings on the enrollment and test audio files, and then uses the average cosine distance between the two sets to determine if the speakers are the same.

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
