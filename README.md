# Accelerated Whisper Inference with Speculative Decoding

This repository implements Speculative Decoding for OpenAI's Whisper model. By using a smaller "draft" model (e.g., `whisper-tiny`) to preview tokens and a larger "main" model (e.g., `whisper-large-v2`) to verify them, we achieve significant speedups without sacrificing accuracy.

## Key Innovation:  Cross-Version Speculative Decoding

This project introduces **cross-version speculative decoding**, enabling Whisper Large-V3 to leverage incompatible draft models (like Tiny or Base) through: 
- **Text-based token remapping**: Maps draft tokens to main vocabulary via decode/encode
- **Dual feature extraction**: Processes audio separately for each model (128 vs 80 Mel bins)
- **Custom verification logic**: Handles tokenizer and architecture differences

This allows using the latest Whisper V3 with existing draft models, achieving 1.24x speedup despite architectural incompatibility.

## Key Results

### Whisper Large-V2 (Same-Version Speculative)

| Model Configuration | Draft Model | Speedup | WER |
| :--- | :--- | :--- | :--- |
| **Large-V2 (Baseline)** | None | 1.0x | 2.4% |
| **Speculative** | `whisper-tiny` | **1.92x** | 2.4% |
| **Speculative** | `whisper-base` | **1.74x** | 2.4% |
| **Speculative** | `distil-large-v2` | **1.36x** | 2.4% |

### Whisper Large-V3 (Cross-Version Speculative)

| Model Configuration | Draft Model | Speedup | WER |
| :--- | : --- | :--- | :--- |
| **Large-V3 (Baseline)** | None | 1.0x | 9.09% |
| **Cross-Version Speculative** | `whisper-tiny` | **1.24x** | 9.09% |

### Beam Search Analysis (Large-V3)

| Beam Size | Latency | WER | vs Greedy |
| :--- | :--- | :--- | :--- |
| 1 (Greedy) | 16.73s | 14.02% | 1.0x |
| 2 | 20.94s | 14.02% | 0.80x |
| 3 | 23.27s | 14.02% | 0.72x |
| 4 | 25.36s | 14.02% | 0.66x |
| 5 | 28.01s | 14.02% | 0.60x |

**Finding**:  Beam search provides no WER improvement on clean audio while increasing latency by 1.67x.

*Note:  Benchmarks conducted on NVIDIA T4 GPU using LibriSpeech validation data.*

## Features

* **Speculative Decoding Engine**: Custom wrapper class for Hugging Face Transformers to handle multi-model inference
* **Cross-Version Support**: Novel token remapping approach enabling Large-V3 to use incompatible draft models
* **Draft Model Support**: Plug-and-play support for `tiny`, `base`, and `distil` variants
* **Comprehensive Benchmarks**:
    * **Speculative vs. Standard**: Visualizes latency vs. accuracy trade-offs
    * **Cross-Version Analysis**: Demonstrates V3 + Tiny compatibility
    * **Beam Search Comparison**: Evaluates beam search strategies
* **Production API**: FastAPI server for batched audio transcription
* **Interactive Notebook**: Jupyter notebook with full implementation and experiments

## Technical Architecture

### Why Whisper Large-V2 for Standard Speculative Decoding? 

Whisper Large-V3 introduced significant architectural changes:
- **Feature Extractor**: 128 Mel frequency bins (vs. 80 in V2)
- **Tokenizer**: Updated vocabulary with 51,865 tokens (vs. 51,864 in V2)
- **Encoder**: Modified positional encodings

Most available draft models (`distil-large-v2`, `tiny`, `base`) were trained with V2's architecture, making them directly compatible only with Large-V2.

### Cross-Version Solution

For Large-V3, we implement: 

1. **Dual Feature Extraction Pipeline**
   ```python
   main_features = FeatureExtractor_V3(audio)   # 128 Mel bins
   draft_features = FeatureExtractor_Tiny(audio) # 80 Mel bins
   ```

2. **Text-Based Token Remapping**
   ```python
   for draft_token in draft_vocabulary:
       text = draft_tokenizer.decode(draft_token)
       main_token = main_tokenizer.encode(text)[0]
       token_map[draft_token] = main_token
   ```
   - 100% coverage:  51,865/51,865 tokens mapped
   - 99.2% exact text matches
   - 0.8% multi-token approximations

3. **Custom Verification Logic**
   - Draft model generates K tokens using 80 Mel bins
   - Tokens remapped to main vocabulary
   - Main model verifies using 128 Mel bins
   - Accept matching tokens, reject mismatches

**Trade-off**: 35% performance penalty vs. same-version (1.24x vs. 1.92x) due to: 
- Dual feature extraction overhead (~10%)
- Token remapping computation (~5%)
- Lower acceptance rate (~40% vs ~60%)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/prathamc25/whisper-speculative-decoding.git
   cd whisper-speculative-decoding
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Run Benchmarks

**Speculative Decoding (Large-V2 + Draft Models)**
```bash
python main.py benchmark --mode speculative --samples 8
```

**Cross-Version Speculative (Large-V3 + Tiny)**
```bash
python main.py benchmark --mode v3 --samples 10
```

**Beam Search Analysis**
```bash
python main.py benchmark --mode beam --samples 20
```

### 2. Start REST API

Launch the production-ready API server:
```bash
python main.py api --port 8000
```

Test the API: 
```bash
curl -X POST "http://localhost:8000/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/audio.wav"
```

### 3. Interactive Notebook

Open `notebook_implementation.ipynb` for:
- Step-by-step implementation walkthrough
- Detailed benchmarks with visualizations
- Cross-version token remapping demonstration
- Beam search analysis

## Project Structure

```
whisper-speculative-decoding/
├── main.py                          # CLI entry point
├── model.py                         # SpeculativeWhisper and SpeculativeWhisperV3 classes
├── benchmark. py                     # Benchmark runners and plotting
├── api.py                          # FastAPI server
├── utils.py                        # Helper functions
├── notebook_implementation.ipynb   # Interactive experiments
├── report.pdf                         # LaTeX report
└── requirements.txt                # Dependencies
```

## Implementation Details

### SpeculativeWhisper Class (Large-V2)

```python
from model import SpeculativeWhisper

# Initialize with draft model
sw = SpeculativeWhisper(
    model_id="openai/whisper-large-v2",
    draft_model_id="openai/whisper-tiny"
)

# Transcribe with speculative decoding
transcriptions, latency = sw.transcribe(
    audio_inputs=["audio1.wav", "audio2.wav"],
    batch_size=1,
    use_speculative=True
)
```

### SpeculativeWhisperV3 Class (Cross-Version)

```python
from model import SpeculativeWhisperV3

# Initialize with cross-version support
sw_v3 = SpeculativeWhisperV3(
    model_id="openai/whisper-large-v3",
    draft_model_id="openai/whisper-tiny"
)
# Automatically builds token mapping (51,865 tokens)

# Transcribe with token remapping
transcriptions, latency = sw_v3.transcribe(
    audio_inputs=["audio1.wav", "audio2.wav"],
    batch_size=1,
    use_speculative=True
)
```

## When to Use Each Approach

### Use Large-V2 + Tiny Speculative When:
- Best speed/accuracy balance required (1.92x speedup)
- V2 accuracy is sufficient for your domain
- Production deployment with cost constraints

### Use Large-V3 + Tiny Cross-Version When:
- Latest V3 accuracy is essential
- No V3-compatible draft models available
- 1.24x speedup justifies implementation complexity

### Use Beam Search When:
- Maximum accuracy required regardless of latency
- Noisy audio with ambiguous interpretations
- Not recommended for clean audio (no WER gain observed)

## Performance Benchmarks

All benchmarks on NVIDIA T4 GPU, FP16 precision, LibriSpeech clean validation set.

**Speedup vs WER Trade-offs:**
- Large-V2 + Tiny: 1.92x faster, 0% WER degradation
- Large-V2 + Base: 1.74x faster, 0% WER degradation
- Large-V3 + Tiny: 1.24x faster, 0% WER degradation
- Beam Search (5): 0.60x slower, 0% WER improvement

**Time Savings (per 10s audio):**
- Large-V2 baseline: 8.05s
- Large-V2 + Tiny: 4.19s (saves 3.86s, 48% reduction)
- Large-V3 baseline: 10.90s
- Large-V3 + Tiny: 8.80s (saves 2.10s, 19% reduction)

## Contributing

Contributions are welcome.  Areas for improvement:
- Training V3-compatible draft models (eliminate remapping overhead)
- Learned token mappings (neural network-based)
- Adaptive lookahead strategies
- Multilingual evaluation
- Noisy audio benchmarks

Please open an issue or submit a pull request. 



## License

This project is licensed under the MIT License.  See LICENSE for details. 

## Acknowledgments

- OpenAI for the Whisper model
- Hugging Face for the Transformers library
- Original speculative decoding papers:  Leviathan et al.  (2023), Chen et al. (2023)