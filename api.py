from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
from model import SpeculativeWhisper, SpeculativeWhisperV3
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Whisper Speculative Decoding API",
    description="Fast speech-to-text transcription using speculative decoding",
    version="1.0.0"
)

# Global model instances (loaded on startup)
v2_model = None
v3_model = None

@app. on_event("startup")
async def load_models():
    global v2_model, v3_model
    
    logger.info("Loading Whisper models...")
    
    # Load V2 + Tiny (faster, good for most use cases)
    v2_model = SpeculativeWhisper(
        model_id="openai/whisper-large-v2",
        draft_model_id="openai/whisper-tiny"
    )
    logger.info("Loaded V2 + Tiny model")
    
    # Optionally load V3 + Tiny (uncomment if needed)
    # v3_model = SpeculativeWhisperV3(
    #     model_id="openai/whisper-large-v3",
    #     draft_model_id="openai/whisper-tiny"
    # )
    # logger.info("Loaded V3 + Tiny model")

@app.get("/")
async def root():
    return {
        "message": "Whisper Speculative Decoding API",
        "endpoints": {
            "POST /transcribe": "Transcribe audio file",
            "GET /health": "Health check",
            "GET /models": "List available models"
        }
    }

@app. get("/health")
async def health_check():
    return {
        "status": "healthy",
        "v2_model_loaded": v2_model is not None,
        "v3_model_loaded":  v3_model is not None
    }

@app.get("/models")
async def list_models():
    return {
        "available_models": [
            {
                "id": "v2",
                "name": "Whisper Large-V2 + Tiny",
                "speedup": "1.92x",
                "wer":  "2.4%",
                "status": "loaded" if v2_model else "not loaded"
            },
            {
                "id": "v3",
                "name": "Whisper Large-V3 + Tiny (Cross-Version)",
                "speedup": "1.24x",
                "wer": "9.09%",
                "status": "loaded" if v3_model else "not loaded"
            }
        ]
    }

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(... ),
    model: str = "v2",
    language: str = "en",
    use_speculative: bool = True,
    task: str = "transcribe"
):
    """
    Transcribe an audio file. 
    
    Parameters:
    - file: Audio file (wav, mp3, flac, etc.)
    - model: Model to use ("v2" or "v3")
    - language: Language code (e.g., "en", "es", "fr")
    - use_speculative: Enable speculative decoding (default: True)
    - task: "transcribe" or "translate"
    """
    
    # Validate model selection
    if model == "v2" and v2_model is None:
        raise HTTPException(status_code=503, detail="V2 model not loaded")
    if model == "v3" and v3_model is None:
        raise HTTPException(status_code=503, detail="V3 model not loaded")
    if model not in ["v2", "v3"]:
        raise HTTPException(status_code=400, detail="Invalid model.  Choose 'v2' or 'v3'")
    
    # Save uploaded file temporarily
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file. filename)[1]) as temp: 
            content = await file.read()
            temp. write(content)
            temp_file = temp.name
        
        logger.info(f"Processing file: {file.filename} with model: {model}")
        
        # Select model
        selected_model = v2_model if model == "v2" else v3_model
        
        # Transcribe
        transcriptions, latency = selected_model.transcribe(
            audio_inputs=[temp_file],
            batch_size=1,
            use_speculative=use_speculative,
            language=language,
            task=task
        )
        
        logger.info(f"Transcription completed in {latency:.2f}s")
        
        return JSONResponse({
            "success": True,
            "transcription": transcriptions[0],
            "metadata": {
                "filename": file.filename,
                "model": model,
                "language": language,
                "speculative_enabled": use_speculative,
                "latency_seconds": round(latency, 3),
                "task": task
            }
        })
    
    except Exception as e: 
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

@app.post("/transcribe/batch")
async def transcribe_batch(
    files: list[UploadFile] = File(...),
    model: str = "v2",
    language: str = "en",
    use_speculative: bool = True,
    task: str = "transcribe"
):
    """
    Transcribe multiple audio files. 
    
    Parameters:
    - files: List of audio files
    - model: Model to use ("v2" or "v3")
    - language: Language code
    - use_speculative: Enable speculative decoding
    - task: "transcribe" or "translate"
    """
    
    # Validate model selection
    if model == "v2" and v2_model is None:
        raise HTTPException(status_code=503, detail="V2 model not loaded")
    if model == "v3" and v3_model is None:
        raise HTTPException(status_code=503, detail="V3 model not loaded")
    
    temp_files = []
    try:
        # Save all uploaded files
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
                content = await file. read()
                temp.write(content)
                temp_files. append(temp.name)
        
        logger.info(f"Processing {len(files)} files with model: {model}")
        
        # Select model
        selected_model = v2_model if model == "v2" else v3_model
        
        # Transcribe batch
        transcriptions, latency = selected_model.transcribe(
            audio_inputs=temp_files,
            batch_size=1,
            use_speculative=use_speculative,
            language=language,
            task=task
        )
        
        logger.info(f"Batch transcription completed in {latency:.2f}s")
        
        # Format results
        results = [
            {
                "filename": files[i].filename,
                "transcription": transcriptions[i]
            }
            for i in range(len(files))
        ]
        
        return JSONResponse({
            "success": True,
            "results": results,
            "metadata":  {
                "num_files": len(files),
                "model": model,
                "language": language,
                "speculative_enabled": use_speculative,
                "total_latency_seconds": round(latency, 3),
                "avg_latency_seconds": round(latency / len(files), 3),
                "task": task
            }
        })
    
    except Exception as e: 
        logger.error(f"Batch transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch transcription failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def run_server(port: int = 8000):
    """
    Start the FastAPI server.
    
    Parameters:
    - port:  Port to run the server on (default: 8000)
    """
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    run_server()