# Accelerated Whisper Inference with Speculative Decoding

This repository implements Speculative Decoding for OpenAI's Whisper model. By using a smaller "draft" model (e.g., `whisper-tiny`) to preview tokens and a larger "main" model (e.g., `whisper-large-v2`) to verify them, this project achieves significant latency reductions while maintaining the exact same Word Error Rate (WER) as the large model.

## Key Results

| Model Configuration | Draft Model | Speedup | WER |
| :--- | :--- | :--- | :--- |
| **Large-V2 (Baseline)** | None | 1.0x | ~2.4% |
| **Speculative** | `whisper-tiny` | **~1.8x** | ~2.4% |
| **Speculative** | `whisper-base` | **~1.5x** | ~2.4% |

*Note: Results may vary by hardware. Benchmarks were conducted on an NVIDIA T4 GPU using LibriSpeech validation data.*

## Features

* **Speculative Decoding Engine:** Custom wrapper class for Hugging Face Transformers to handle multi-model inference.
* **Draft Model Support:** Plug-and-play support for `tiny`, `base`, and `distil` variants.
* **Comprehensive Benchmarks:**
    * **Speculative vs. Standard:** Visualizes latency vs. accuracy trade-offs.
    * **Beam Search Analysis:** Compares speculative decoding against high-accuracy beam search strategies.
* **Production API:** Includes a FastAPI server for batched audio transcription.

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/prathamc25/whisper-speculative-decoding.git](https://github.com/prathamc25/whisper-speculative-decoding.git)
    cd whisper-speculative-decoding
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run Benchmarks
Run the full suite of experiments to generate performance plots.

To run the Speculative Decoding benchmark (comparing different draft models):
```bash
python main.py benchmark --mode speculative

```

To run the Beam Search benchmark (comparing beam search across model sizes):

```bash
python main.py benchmark --mode beam

```

### 2. Start REST API

Launch the production-ready API server:

```bash
python main.py api --port 8000

```

To test the API, you can use `curl`:

```bash
curl -X POST "http://localhost:8000/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/audio.wav"

```

## Project Structure

* `main.py`: The entry point for the command-line interface. Handles argument parsing and routing to benchmarks or API.
* `model.py`: Contains the `SpeculativeWhisper` class. This handles the loading of main/draft models and the transcription logic.
* `benchmark.py`: Contains the logic for running inference loops on the LibriSpeech dataset and generating performance plots.
* `api.py`: The FastAPI server implementation.
* `utils.py`: Helper functions for dependency checking and management.
* `exploration.ipynb`: A Jupyter Notebook containing initial experiments, visualizations, and interactive tests.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request if you have suggestions for optimizing draft model selection or improving inference speed.

## License

This project is licensed under the MIT License.

```

```
