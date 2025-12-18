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

        print(f"   [Init] Loading Main:  {model_id}...")
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

        self.processor = AutoProcessor. from_pretrained(model_id)

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
                   batch_size:  int = 1,
                   use_speculative: bool = True,
                   language: str = "en",
                   task: str = "transcribe",
                   **generate_kwargs):

        # Safety Check:  Beam Search prevents Speculative Decoding
        if generate_kwargs.get("num_beams", 1) > 1 and use_speculative:
            use_speculative = False

        raw_audio = self._load_audio(audio_inputs)
        transcriptions = []
        total_inference_time = 0

        for i in range(0, len(raw_audio), batch_size):
            batch_audio = raw_audio[i :  i + batch_size]

            inputs = self.processor(
                batch_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )
            input_features = inputs. input_features. to(self.device, dtype=self.dtype)

            gen_args = {
                "input_features": input_features,
                "max_new_tokens": 400,
                "language": language,
                "task":  task,
                **generate_kwargs
            }

            if use_speculative and self.draft_model:
                gen_args["assistant_model"] = self.draft_model

            if self.device == "cuda":  torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                generated_ids = self.main_model.generate(**gen_args)

            if self.device == "cuda":  torch.cuda.synchronize()
            total_inference_time += (time. time() - start_time)

            batch_transcripts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            transcriptions. extend(batch_transcripts)

        return transcriptions, total_inference_time


class SpeculativeWhisperV3:
    def __init__(self,
                 model_id:  str = "openai/whisper-large-v3",
                 draft_model_id: Optional[str] = None,
                 device: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None):

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch_dtype if torch_dtype else (torch.float16 if self.device == "cuda" else torch.float32)

        print(f"   [Init] Loading Main:   {model_id}...")
        self.main_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        ).to(self.device)
        
        self.main_processor = AutoProcessor.from_pretrained(model_id)

        self.draft_model = None
        self.draft_processor = None
        self.use_token_remapping = False
        
        if draft_model_id: 
            print(f"   [Init] Loading Draft:  {draft_model_id}...")
            self.draft_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                draft_model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="sdpa",
            ).to(self.device)
            
            self.draft_processor = AutoProcessor.from_pretrained(draft_model_id)
            
            if self._tokenizers_differ():
                print(f"   [Warning] Tokenizers differ between models.  Enabling token remapping.")
                self. use_token_remapping = True
                self._build_token_mapping()

    def _tokenizers_differ(self) -> bool:
        main_vocab_size = self.main_model.config.vocab_size
        draft_vocab_size = self. draft_model.config.vocab_size
        return main_vocab_size != draft_vocab_size

    def _build_token_mapping(self):
        print(f"   [Init] Building token mapping...")
        
        main_tokenizer = self.main_processor.tokenizer
        draft_tokenizer = self.draft_processor.tokenizer
        
        self.token_map = {}
        
        for draft_id in range(self.draft_model.config.vocab_size):
            try:
                draft_text = draft_tokenizer.decode([draft_id], skip_special_tokens=False)
                main_ids = main_tokenizer.encode(draft_text, add_special_tokens=False)
                
                if main_ids:
                    self. token_map[draft_id] = main_ids[0]
                else:
                    self.token_map[draft_id] = main_tokenizer.unk_token_id
            except:
                self.token_map[draft_id] = main_tokenizer. unk_token_id
        
        print(f"   [Init] Mapped {len(self.token_map)} tokens")

    def _remap_draft_tokens(self, draft_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_token_remapping:
            return draft_ids
        
        remapped = []
        for token_id in draft_ids. cpu().numpy().flatten():
            remapped.append(self.token_map.get(int(token_id), self.main_processor.tokenizer.unk_token_id))
        
        return torch.tensor(remapped, device=draft_ids.device, dtype=draft_ids.dtype).reshape(draft_ids.shape)

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
                   batch_size:  int = 1,
                   use_speculative: bool = True,
                   language: str = "en",
                   task:  str = "transcribe",
                   **generate_kwargs):

        if generate_kwargs.get("num_beams", 1) > 1 and use_speculative: 
            use_speculative = False

        raw_audio = self._load_audio(audio_inputs)
        transcriptions = []
        total_inference_time = 0

        for i in range(0, len(raw_audio), batch_size):
            batch_audio = raw_audio[i :   i + batch_size]

            main_inputs = self.main_processor(
                batch_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )
            main_input_features = main_inputs.input_features.to(self.device, dtype=self.dtype)

            gen_args = {
                "max_new_tokens": 400,
                "language": language,
                "task":   task,
                **generate_kwargs
            }

            if use_speculative and self.draft_model and self.use_token_remapping:
                draft_inputs = self. draft_processor(
                    batch_audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True
                )
                draft_input_features = draft_inputs.input_features.to(self.device, dtype=self.dtype)
                
                if self.device == "cuda":   torch.cuda.synchronize()
                start_time = time.time()
                
                generated_ids = self._speculative_generate_with_remapping(
                    main_input_features=main_input_features,
                    draft_input_features=draft_input_features,
                    max_new_tokens=gen_args["max_new_tokens"],
                    language=gen_args["language"],
                    task=gen_args["task"]
                )
                
                if self.device == "cuda":  torch.cuda.synchronize()
                total_inference_time += (time.time() - start_time)
            
            elif use_speculative and self.draft_model:
                gen_args["assistant_model"] = self.draft_model
                gen_args["input_features"] = main_input_features
                
                if self. device == "cuda":  torch.cuda.synchronize()
                start_time = time.time()

                with torch.no_grad():
                    generated_ids = self.main_model.generate(**gen_args)

                if self.device == "cuda": torch.cuda.synchronize()
                total_inference_time += (time.time() - start_time)
            
            else:
                gen_args["input_features"] = main_input_features
                
                if self.device == "cuda": torch.cuda.synchronize()
                start_time = time. time()

                with torch.no_grad():
                    generated_ids = self.main_model. generate(**gen_args)

                if self.device == "cuda":  torch.cuda.synchronize()
                total_inference_time += (time.time() - start_time)

            batch_transcripts = self.main_processor.batch_decode(generated_ids, skip_special_tokens=True)
            transcriptions. extend(batch_transcripts)

        return transcriptions, total_inference_time

    def _speculative_generate_with_remapping(
        self,
        main_input_features: torch.Tensor,
        draft_input_features: torch.Tensor,
        max_new_tokens: int = 400,
        language: str = "en",
        task:  str = "transcribe",
        num_draft_tokens: int = 3,
        **kwargs
    ):
        forced_decoder_ids = self.main_processor.get_decoder_prompt_ids(
            language=language, 
            task=task
        )
        
        initial_tokens = [self.main_model.config.decoder_start_token_id]
        if forced_decoder_ids:
            initial_tokens.extend([token_id for _, token_id in forced_decoder_ids])
        
        decoder_input_ids = torch.tensor([initial_tokens], device=self.device)
        
        with torch.no_grad():
            main_encoder_outputs = self.main_model.get_encoder()(main_input_features)
            draft_encoder_outputs = self.draft_model.get_encoder()(draft_input_features)
            
            for step in range(max_new_tokens):
                draft_next_tokens = []
                current_draft_ids = decoder_input_ids. clone()
                
                for k in range(num_draft_tokens):
                    draft_outputs = self.draft_model(
                        encoder_outputs=draft_encoder_outputs,
                        decoder_input_ids=current_draft_ids,
                    )
                    draft_next_token = draft_outputs.logits[: , -1, :].argmax(dim=-1, keepdim=True)
                    draft_next_tokens.append(draft_next_token)
                    current_draft_ids = torch.cat([current_draft_ids, draft_next_token], dim=1)
                
                draft_sequence = torch.cat(draft_next_tokens, dim=1)
                remapped_draft = self._remap_draft_tokens(draft_sequence)
                
                candidate = torch.cat([decoder_input_ids, remapped_draft], dim=1)
                
                main_outputs = self.main_model(
                    encoder_outputs=main_encoder_outputs,
                    decoder_input_ids=candidate,
                )
                
                main_next_tokens = main_outputs.logits. argmax(dim=-1)
                
                num_accepted = 0
                start_pos = decoder_input_ids.shape[1] - 1
                
                for k in range(num_draft_tokens):
                    main_pred = main_next_tokens[0, start_pos + k]
                    draft_token = remapped_draft[0, k]
                    
                    if main_pred == draft_token: 
                        num_accepted += 1
                    else:
                        decoder_input_ids = torch.cat([
                            decoder_input_ids, 
                            main_pred.unsqueeze(0).unsqueeze(0)
                        ], dim=1)
                        break
                else:
                    if num_accepted > 0:
                        decoder_input_ids = torch.cat([
                            decoder_input_ids,
                            remapped_draft[: , : num_accepted]
                        ], dim=1)
                    else:
                        next_token = main_next_tokens[: , start_pos]. unsqueeze(1)
                        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
                
                if decoder_input_ids[0, -1]. item() == self.main_processor.tokenizer.eos_token_id:
                    break
                    
                if decoder_input_ids.shape[1] >= max_new_tokens + len(initial_tokens):
                    break
        
        return decoder_input_ids