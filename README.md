# Accelerated Whisper Inference with Speculative Decoding

This repository implements Speculative Decoding for OpenAI's Whisper model.  By using a smaller "draft" model (e.g., `whisper-tiny`) to preview tokens and a larger "main" model (e. g., `whisper-large-v2`) to verify them, we achieve significant speedups without sacrificing accuracy.

## Key Innovation:   Cross-Version Speculative Decoding

This project introduces **cross-version speculative decoding**, enabling Whisper Large-V3 to leverage incompatible draft models (like Tiny or Base) through:  
- **Text-based token remapping**:  Maps draft tokens to main vocabulary via decode/encode
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
| :--- | :  --- | :--- | :--- |
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

**Finding**:   Beam search provides no WER improvement on clean audio while increasing latency by 1.67x. 

*Note:   Benchmarks conducted on NVIDIA T4 GPU using LibriSpeech validation data.*

## Features

* **Speculative Decoding Engine**:  Custom wrapper class for Hugging Face Transformers to handle multi-model inference
* **Cross-Version Support**: Novel token remapping approach enabling Large-V3 to use incompatible draft models
* **Draft Model Support**: Plug-and-play support for `tiny`, `base`, and `distil` variants
* **Comprehensive Benchmarks**:
    * **Speculative vs. Standard**:  Visualizes latency vs. accuracy trade-offs
    * **Cross-Version Analysis**: Demonstrates V3 + Tiny compatibility
    * **Beam Search Comparison**: Evaluates beam search strategies
* **Production API**: FastAPI server for batched audio transcription
* **Interactive Notebooks**: Two Jupyter notebooks with full implementation and experiments

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

### 1. Interactive Notebooks (Recommended for Learning)

We provide two comprehensive Jupyter notebooks for hands-on exploration:

#### notebook_implementation.ipynb - Large-V2 Speculative Decoding

**What's included:**
- Complete `SpeculativeWhisper` class implementation
- Whisper Large-V2 + Tiny/Base/Distil benchmarks
- Beam search analysis across model sizes
- Visualizations and performance plots
- Step-by-step explanations

**Run on Google Colab:**
```bash
# Open the notebook in Colab
https://colab.research.google.com/github/prathamc25/whisper-speculative-decoding/blob/main/notebook_implementation.ipynb
```

**Or run locally:**
```bash
jupyter notebook notebook_implementation.ipynb
```

**Notebook structure:**
1. Install dependencies
2. Define `SpeculativeWhisper` class with helper functions
3. Benchmark:  Speculative decoding with different draft models (8 samples)
4. Benchmark: Beam search analysis (8 samples)
5. Visualizations and results

**Expected runtime:** 10-15 minutes on T4 GPU

---

#### notebook_implementation_tiny_largev3.ipynb - Cross-Version Speculative Decoding

**What's included:**
- Complete `SpeculativeWhisperV3` class with token remapping
- Dual feature extraction implementation
- Large-V3 + Tiny cross-version benchmark
- Token mapping verification
- Detailed performance statistics

**Run on Google Colab:**
```bash
# Open the notebook in Colab
https://colab.research.google.com/github/prathamc25/whisper-speculative-decoding/blob/main/notebook_implementation_tiny_largev3.ipynb
```

**Or run locally:**
```bash
jupyter notebook notebook_implementation_tiny_largev3.ipynb
```

**Notebook structure:**
1. Install dependencies
2. Define `SpeculativeWhisperV3` class with token remapping logic
3. Multi-sample test:  Large-V3 + Tiny (10-500 samples configurable)
4. Detailed metrics:  WER, CER, speedup, exact match rate
5. Token mapping statistics

**Expected runtime:** 
- 10 samples: 2-3 minutes on T4 GPU
- 500 samples: 45-60 minutes on T4 GPU

---

**Which notebook should you use?**

| Use Case | Notebook |
|----------|----------|
| Learn standard speculative decoding | `notebook_implementation.ipynb` |
| Understand V2 architecture | `notebook_implementation.ipynb` |
| Best speedup results (1.92x) | `notebook_implementation.ipynb` |
| Cross-version compatibility | `notebook_implementation_tiny_largev3.ipynb` |
| Latest Whisper V3 features | `notebook_implementation_tiny_largev3.ipynb` |
| Token remapping deep dive | `notebook_implementation_tiny_largev3.ipynb` |

### 2. Command-Line Benchmarks

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

### 3. Production API

Launch the REST API server:
```bash
python main.py api --port 8000
```

**Test single file transcription:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
     -F "file=@audio.wav" \
     -F "model=v2" \
     -F "use_speculative=true"
```

**Test batch transcription:**
```bash
curl -X POST "http://localhost:8000/transcribe/batch" \
     -F "files=@audio1.wav" \
     -F "files=@audio2.wav" \
     -F "model=v2"
```

**API endpoints:**
- `GET /` - API documentation
- `GET /health` - Health check
- `GET /models` - List available models
- `POST /transcribe` - Single file transcription
- `POST /transcribe/batch` - Batch transcription

**Supported parameters:**
- `model`: "v2" (Large-V2 + Tiny) or "v3" (Large-V3 + Tiny cross-version)
- `language`: Language code (e.g., "en", "es", "fr")
- `use_speculative`: Enable/disable speculative decoding (default: true)
- `task`: "transcribe" or "translate"

## Project Structure

```
whisper-speculative-decoding/
├── main.py                                    # CLI entry point
├── model.py                                   # SpeculativeWhisper and SpeculativeWhisperV3 classes
├── benchmark.  py                               # Benchmark runners and plotting
├── api.py                                    # FastAPI server
├── utils.py                                  # Helper functions
├── notebook_implementation.ipynb             # Large-V2 interactive notebook
├── notebook_implementation_tiny_largev3.ipynb # Large-V3 cross-version notebook
├── report/                                   # LaTeX report
│   ├── whisper_speculative_decoding_report.tex
│   ├── compile. sh
│   └── README.md
└── requirements.txt                          # Dependencies
```

## Notebook Features

Both notebooks include: 

- **Self-contained implementations**: Run directly in Colab without cloning the repo
- **Interactive experimentation**: Modify hyperparameters and see results instantly
- **Detailed explanations**:  Markdown cells explaining each step
- **Performance metrics**: WER, CER, latency, speedup calculations
- **Clean outputs**: Formatted tables and statistics

**Tips for notebook usage:**

1. **Start with small samples**: Use 8-10 samples for quick testing
2. **Scale up gradually**:  Increase to 50-500 samples for publication-quality results
3. **Monitor GPU memory**: Large-V3 + Tiny requires ~6GB VRAM
4. **Save checkpoints**: Notebooks auto-save results for analysis

## Technical Deep Dive

### Cross-Version Token Remapping Algorithm

The key innovation enabling V3 + Tiny compatibility: 

```python
def _build_token_mapping(self):
    """Build mapping from draft vocabulary to main vocabulary"""
    for draft_token_id in draft_vocabulary:
        # Decode draft token to text
        text = draft_tokenizer.decode([draft_token_id])
        
        # Re-encode with main tokenizer
        main_token_ids = main_tokenizer.encode(text)
        
        # Map to first token (handles multi-token cases)
        token_map[draft_token_id] = main_token_ids[0]
```

**Coverage achieved**:  51,865/51,865 tokens (100%)

### Dual Feature Extraction

```python
# Process audio separately for each model
main_features = main_processor(audio)   # 128 Mel bins for V3
draft_features = draft_processor(audio) # 80 Mel bins for Tiny

# Each model receives features in its expected format
main_output = main_model(main_features)
draft_output = draft_model(draft_features)
```

### Performance Trade-offs

| Approach | Speedup | Overhead | Best For |
|----------|---------|----------|----------|
| V2 + Tiny | 1.92x | Minimal | Production (fastest) |
| V2 + Base | 1.74x | Low | Balanced |
| V3 + Tiny | 1.24x | Moderate | Latest V3 features |
| Beam Search | 0.60x | High | Maximum accuracy |

## When to Use Each Approach

### Use Large-V2 + Tiny Speculative When:
- Best speed/accuracy balance required (1.92x speedup)
- V2 accuracy is sufficient for your domain
- Production deployment with cost constraints
- Quick experimentation needed

### Use Large-V3 + Tiny Cross-Version When: 
- Latest V3 accuracy is essential
- No V3-compatible draft models available
- 1.24x speedup justifies implementation complexity
- Exploring cutting-edge techniques

### Use Beam Search When:
- Maximum accuracy required regardless of latency
- Noisy audio with ambiguous interpretations
- Not recommended for clean audio (no WER gain observed)

## Contributing

Contributions are welcome.   Areas for improvement:
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
- Original speculative decoding papers:   Leviathan et al.   (2023), Chen et al.  (2023)

## Frequently Asked Questions

**Q: Which notebook should I start with?**  
A: Start with `notebook_implementation.ipynb` for standard speculative decoding (V2). Move to `notebook_implementation_tiny_largev3.ipynb` if you need V3 features.

**Q: Can I run the notebooks locally?**  
A: Yes, but you'll need a CUDA GPU with at least 6GB VRAM.  Google Colab (free tier) works well.

**Q: How long does a full benchmark take?**  
A:  8 samples: 5-10 minutes.  500 samples: 45-60 minutes on T4 GPU.

**Q: Does this work with other languages?**  
A: Yes, Whisper supports 99 languages.  Set `language="es"` (or any ISO code) in the transcribe call.

**Q: Can I use this in production?**  
A: Yes, the FastAPI server is production-ready. Use Large-V2 + Tiny for best performance (1.92x speedup).

**Q: Why is V3 + Tiny slower than V2 + Tiny?**  
A: Token remapping and dual feature extraction add ~35% overhead. Still 1.24x faster than V3 baseline.

**Q: Do you support batch processing?**  
A: Yes, via the API (`/transcribe/batch`) or by passing multiple files to `transcribe()`.

**Q: What if I want to train my own draft model?**  
A: You'll need to match the main model's architecture (Mel bins, tokenizer). See "Contributing" section for guidance. 