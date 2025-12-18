import torch
import time
import librosa
import numpy as np
from typing import List, Union, Optional
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

class SpeculativeWhisper:
    def __init__(self,
                 model_id: str = "openai/whisper-large-v2",
                 draft_model_id: Optional[str] = None,
                 device: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None):

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch_dtype if torch_dtype else (torch.float16 if self.device == "cuda" else torch.float32)

        print(f"   [Init] Loading Main: {model_id}...")
        self.main_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        ).to(self.device)

        self.draft_model = None
        if draft_model_id:
            print(f"   [Init] Loading Draft: {draft_model_id}...")
            self.draft_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                draft_model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="sdpa",
            ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

    def _load_audio(self, audio_paths: Union[List[str], List[np.ndarray]]) -> List[np.ndarray]:
        audio_arrays = []
        for audio in audio_paths:
            if isinstance(audio, str):
                array, _ = librosa.load(audio, sr=16000)
                audio_arrays.append(array)
            elif isinstance(audio, np.ndarray):
                audio_arrays.append(audio)
            else:
                raise ValueError("Audio must be a file path or numpy array.")
        return audio_arrays

    def transcribe(self,
                   audio_inputs: Union[List[str], List[np.ndarray]],
                   batch_size: int = 1,
                   use_speculative: bool = True,
                   language: str = "en",
                   task: str = "transcribe",
                   **generate_kwargs):

        # Safety Check: Beam Search prevents Speculative Decoding
        if generate_kwargs.get("num_beams", 1) > 1 and use_speculative:
            use_speculative = False

        raw_audio = self._load_audio(audio_inputs)
        transcriptions = []
        total_inference_time = 0

        for i in range(0, len(raw_audio), batch_size):
            batch_audio = raw_audio[i : i + batch_size]

            inputs = self.processor(
                batch_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )
            input_features = inputs.input_features.to(self.device, dtype=self.dtype)

            gen_args = {
                "input_features": input_features,
                "max_new_tokens": 400,
                "language": language,
                "task": task,
                **generate_kwargs
            }

            if use_speculative and self.draft_model:
                gen_args["assistant_model"] = self.draft_model

            if self.device == "cuda": torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                generated_ids = self.main_model.generate(**gen_args)

            if self.device == "cuda": torch.cuda.synchronize()
            total_inference_time += (time.time() - start_time)

            batch_transcripts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            transcriptions.extend(batch_transcripts)

        return transcriptions, total_inference_time