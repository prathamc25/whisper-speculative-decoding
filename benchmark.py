import torch
import gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import evaluate
from model import SpeculativeWhisper

def plot_dashboard(results, title_suffix=""):
    df = pd.DataFrame(results)
    baseline = df. iloc[0]["latency"]
    df["speedup"] = baseline / df["latency"]

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2])

    ax1 = fig.add_subplot(gs[0])
    bars = ax1.barh(df["config"], df["latency"], color=df["color"], alpha=0.9)
    ax1.set_title(f"Inference Latency {title_suffix} (Lower is Better)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Latency (seconds)", fontsize=11)
    ax1.invert_yaxis()
    for bar in bars:
        ax1.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2, f'{bar.get_width():.2f}s', va='center')

    ax2 = fig.add_subplot(gs[1])
    sns.scatterplot(data=df, x="latency", y="wer", s=300, hue="config", palette=dict(zip(df["config"], df["color"])), legend=False, ax=ax2, edgecolor="black")
    for i, row in df.iterrows():
        ax2.text(row["latency"], row["wer"] + (max(df["wer"])*0.01), row["config"]. split('\n')[0], ha='center', fontsize=10)
    ax2.set_title("Efficiency: Accuracy vs Speed", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Latency (s) ->", fontsize=11)
    ax2.set_ylabel("WER ->", fontsize=11)

    plt.tight_layout()
    plt.savefig(f"benchmark_{title_suffix. strip('()').lower().replace(' ', '_')}.png")
    print(f"Plot saved to benchmark_{title_suffix.strip('()').lower().replace(' ', '_')}.png")

def run_speculative_benchmark(num_samples=8):
    configs = [
        {"name": "Baseline (Large-V2)", "main": "openai/whisper-large-v2", "draft": None, "color": "#34495e"},
        {"name": "Draft:  Tiny", "main": "openai/whisper-large-v2", "draft": "openai/whisper-tiny", "color": "#e74c3c"},
        {"name": "Draft: Base", "main": "openai/whisper-large-v2", "draft": "openai/whisper-base", "color": "#f1c40f"},
        {"name": "Draft: Distil-Large", "main": "openai/whisper-large-v2", "draft": "distil-whisper/distil-large-v2", "color": "#2ecc71"}
    ]
    _run_loop(configs, num_samples, "Speculative")

def run_beam_benchmark(num_samples=8):
    configs = [
        {"name": "Tiny (Beam 5)", "main": "openai/whisper-tiny", "draft": None, "color": "#e74c3c", "beams": 5},
        {"name": "Base (Beam 5)", "main": "openai/whisper-base", "draft": None, "color": "#f1c40f", "beams": 5},
        {"name": "Large-V2 (Beam 5)", "main": "openai/whisper-large-v2", "draft": None, "color": "#34495e", "beams":  5},
    ]
    _run_loop(configs, num_samples, "Beam Search")

def _run_loop(configs, num_samples, mode_name):
    print(f"Loading {num_samples} samples from LibriSpeech...")
    dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)
    samples = list(dataset. take(num_samples))
    audio_inputs = [x["audio"]["array"] for x in samples]
    references = [x["text"] for x in samples]
    wer_metric = evaluate.load("wer")
    
    results = []
    print(f"\n=== STARTING {mode_name.upper()} BENCHMARK ===")
    
    for cfg in configs:
        print(f"\nRunning Config: {cfg['name']}")
        sw = SpeculativeWhisper(model_id=cfg["main"], draft_model_id=cfg.get("draft"))
        
        use_spec = True if cfg.get("draft") else False
        num_beams = cfg.get("beams", 1)
        
        preds, duration = sw.transcribe(
            audio_inputs, 
            batch_size=1, 
            use_speculative=use_spec,
            num_beams=num_beams
        )
        
        wer = wer_metric.compute(predictions=preds, references=references)
        results.append({"config": cfg["name"], "latency": duration, "wer": wer, "color": cfg["color"]})
        
        del sw
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== RESULTS ===")
    results_df = pd.DataFrame(results)[["config", "latency", "wer"]]
    print(results_df.to_string(index=False))
    
    plot_dashboard(results, title_suffix=f"({mode_name})")

def run_v3_benchmark(num_samples=10):
    from model import SpeculativeWhisperV3
    from jiwer import wer
    
    print(f"Loading {num_samples} samples from LibriSpeech...")
    dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)
    samples = list(dataset.take(num_samples))
    audio_inputs = [x["audio"]["array"] for x in samples]
    references = [x["text"]. lower().strip() for x in samples]
    
    print("\n=== STARTING LARGE-V3 CROSS-VERSION BENCHMARK ===")
    
    sw_v3 = SpeculativeWhisperV3(
        model_id="openai/whisper-large-v3",
        draft_model_id="openai/whisper-tiny"
    )
    
    print("\nRunning BASELINE (Large-V3 only)...")
    baseline_preds, baseline_time = sw_v3.transcribe(
        audio_inputs,
        batch_size=1,
        use_speculative=False
    )
    baseline_preds_norm = [p.lower().strip() for p in baseline_preds]
    
    print("\nRunning SPECULATIVE (Large-V3 + Tiny)...")
    spec_preds, spec_time = sw_v3.transcribe(
        audio_inputs,
        batch_size=1,
        use_speculative=True
    )
    spec_preds_norm = [p.lower().strip() for p in spec_preds]
    
    baseline_wer = wer(references, baseline_preds_norm)
    spec_wer = wer(references, spec_preds_norm)
    speedup = baseline_time / spec_time
    
    print("\n" + "="*70)
    print("LARGE-V3 CROSS-VERSION RESULTS")
    print("="*70)
    print(f"\nBASELINE (Large-V3):")
    print(f"  Latency: {baseline_time:.2f}s")
    print(f"  WER:      {baseline_wer:.4f}")
    
    print(f"\nSPECULATIVE (V3 + Tiny):")
    print(f"  Latency:  {spec_time:.2f}s")
    print(f"  WER:     {spec_wer:.4f}")
    print(f"  Speedup:  {speedup:.2f}x")
    print("="*70)
    
    del sw_v3
    gc. collect()
    torch.cuda.empty_cache()