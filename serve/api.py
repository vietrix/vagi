"""
vAGI Streaming API Server (OpenAI-Compatible).

Production-ready FastAPI server with:
- OpenAI-compatible /v1/chat/completions endpoint
- Server-Sent Events (SSE) streaming
- Thinking/Content chunk distinction
- CORS middleware for frontend integration
- Async request handling

Usage:
    # Start server
    uvicorn serve.api:app --host 0.0.0.0 --port 8000 --reload

    # With custom model
    MODEL_PATH=./models/vagi-7b uvicorn serve.api:app --host 0.0.0.0 --port 8000

    # Production with multiple workers
    uvicorn serve.api:app --host 0.0.0.0 --port 8000 --workers 4

API Example:
    curl -X POST http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "vagi", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
"""

import os
import sys
import json
import time
import uuid
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator, Union
from dataclasses import dataclass, field
from enum import Enum

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try importing model components
try:
    from core.agi import AGIModel
    from core.agi.config import load_agi_small_config, load_agi_tiny_config
    from core.nlp import BytePairTokenizer
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# =============================================================================
# Pydantic Models (OpenAI-Compatible)
# =============================================================================

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "vagi"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=32768)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class DeltaContent(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None
    is_thought: bool = False  # Custom field for thinking/content distinction


class StreamChoice(BaseModel):
    index: int
    delta: DeltaContent
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


# =============================================================================
# Model Manager
# =============================================================================

class ModelManager:
    """Manages model loading and inference."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_id = "vagi"
        self.is_vagi = False
        self._lock = asyncio.Lock()

    async def load_model(self, model_path: Optional[str] = None):
        """Load model asynchronously."""
        async with self._lock:
            if self.model is not None:
                return

            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

            print(f"Loading model on {self.device}...")

            model_path = model_path or os.environ.get("MODEL_PATH", "checkpoints/model.pt")

            # Try loading vAGI model
            if model_path.endswith('.pt') and MODEL_AVAILABLE and os.path.exists(model_path):
                await self._load_vagi_model(model_path)
            elif TRANSFORMERS_AVAILABLE and os.path.exists(model_path):
                await self._load_hf_model(model_path)
            else:
                print("Warning: No model loaded. Using mock responses.")
                self.model = None

    async def _load_vagi_model(self, model_path: str):
        """Load vAGI internal model."""
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        config = ckpt.get('config', load_agi_tiny_config())
        self.model = AGIModel(config)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model = self.model.to(self.device).eval()

        self.tokenizer = BytePairTokenizer(vocab_size=config.vocab_size)
        if 'tokenizer_vocab' in ckpt:
            self.tokenizer.vocab = ckpt['tokenizer_vocab']
            self.tokenizer.merges = [tuple(m) for m in ckpt.get('tokenizer_merges', [])]
            self.tokenizer.inverse_vocab = {v: k for k, v in self.tokenizer.vocab.items()}

        self.is_vagi = True
        self.model_id = "vagi-" + config.hidden_size.__str__()
        print(f"vAGI model loaded: {self.model_id}")

    async def _load_hf_model(self, model_path: str):
        """Load HuggingFace model."""
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=str(self.device) if self.device.type == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.model_id = os.path.basename(model_path)
        print(f"HuggingFace model loaded: {self.model_id}")

    def _build_prompt(self, messages: List[ChatMessage]) -> str:
        """Build prompt from messages."""
        prompt_parts = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                prompt_parts.append(f"System: {msg.content}\n")
            elif msg.role == MessageRole.USER:
                prompt_parts.append(f"User: {msg.content}\n")
            elif msg.role == MessageRole.ASSISTANT:
                prompt_parts.append(f"Assistant: {msg.content}\n")

        prompt_parts.append("Assistant: <think>")
        return "".join(prompt_parts)

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncGenerator[tuple[str, bool], None]:
        """
        Generate tokens with streaming.

        Yields:
            (token_text, is_thought) tuples
        """
        prompt = self._build_prompt(messages)

        if self.model is None:
            # Mock streaming for testing
            async for token, is_thought in self._mock_stream():
                yield token, is_thought
            return

        if self.is_vagi:
            async for token, is_thought in self._generate_vagi_stream(prompt, max_tokens, temperature):
                yield token, is_thought
        else:
            async for token, is_thought in self._generate_hf_stream(prompt, max_tokens, temperature):
                yield token, is_thought

    async def _mock_stream(self) -> AsyncGenerator[tuple[str, bool], None]:
        """Mock streaming for testing without model."""
        # Thinking phase
        thinking_text = "Let me analyze this step by step.\n1. First, I need to understand the question.\n2. Then, I'll formulate a response."
        for char in thinking_text:
            yield char, True
            await asyncio.sleep(0.02)

        yield "</think>\n", True

        # Response phase
        response_text = "Hello! I'm vAGI, a reasoning AI system. How can I help you today?"
        for word in response_text.split():
            yield word + " ", False
            await asyncio.sleep(0.05)

    async def _generate_vagi_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[tuple[str, bool], None]:
        """Stream generation with vAGI model."""
        ids = self.tokenizer.encode(prompt, max_length=512)
        generated = ids.copy()
        in_thinking = True

        with torch.no_grad():
            for _ in range(max_tokens):
                x = torch.tensor([generated[-512:]], dtype=torch.long, device=self.device)
                out = self.model(input_ids=x, mode='inference')
                logits = out.get('text_logits')

                if logits is None:
                    break

                # Sample next token
                if temperature > 0:
                    probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
                    next_tok = torch.multinomial(probs, 1).item()
                else:
                    next_tok = logits[0, -1].argmax().item()

                if next_tok == 0:  # EOS
                    break

                generated.append(next_tok)
                token_text = self.tokenizer.decode([next_tok])

                # Check for </think> to switch from thinking to content
                if "</think>" in token_text:
                    in_thinking = False

                yield token_text, in_thinking
                await asyncio.sleep(0.01)  # Small delay for streaming effect

    async def _generate_hf_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[tuple[str, bool], None]:
        """Stream generation with HuggingFace model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        input_length = inputs['input_ids'].shape[1]
        in_thinking = True

        # Generate token by token
        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                new_token_id = outputs[0, -1].item()
                if new_token_id == self.tokenizer.eos_token_id:
                    break

                token_text = self.tokenizer.decode([new_token_id])

                # Check for </think>
                if "</think>" in token_text:
                    in_thinking = False

                yield token_text, in_thinking

                # Update inputs for next iteration
                inputs = {'input_ids': outputs, 'attention_mask': torch.ones_like(outputs)}
                await asyncio.sleep(0.01)

    async def generate_complete(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> tuple[str, int, int]:
        """
        Generate complete response (non-streaming).

        Returns:
            (response_text, prompt_tokens, completion_tokens)
        """
        full_response = []
        async for token, _ in self.generate_stream(messages, max_tokens, temperature):
            full_response.append(token)

        response_text = "".join(full_response)
        prompt_tokens = len(self._build_prompt(messages).split())  # Rough estimate
        completion_tokens = len(response_text.split())

        return response_text, prompt_tokens, completion_tokens


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="vAGI API",
    description="OpenAI-compatible API for vAGI reasoning model with streaming support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model manager
model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    await model_manager.load_model()


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "vAGI API",
        "version": "1.0.0",
        "model": model_manager.model_id,
        "status": "ready" if model_manager.model is not None else "no_model",
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_manager.model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vagi",
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Supports both streaming and non-streaming responses.
    Streaming responses include is_thought field to distinguish
    between thinking and content chunks.
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            generate_sse_stream(request, request_id, created),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        # Non-streaming response
        response_text, prompt_tokens, completion_tokens = await model_manager.generate_complete(
            messages=request.messages,
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature,
        )

        return ChatCompletionResponse(
            id=request_id,
            created=created,
            model=model_manager.model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


async def generate_sse_stream(
    request: ChatCompletionRequest,
    request_id: str,
    created: int,
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events stream.

    Each chunk includes is_thought field to distinguish
    thinking content from response content.
    """
    # Send role delta first
    initial_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model_manager.model_id,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaContent(role="assistant"),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    # Stream content
    async for token, is_thought in model_manager.generate_stream(
        messages=request.messages,
        max_tokens=request.max_tokens or 2048,
        temperature=request.temperature,
    ):
        chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model_manager.model_id,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaContent(content=token, is_thought=is_thought),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Send final chunk with finish_reason
    final_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model_manager.model_id,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaContent(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "device": str(model_manager.device) if model_manager.device else None,
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    print(f"Starting vAGI API server on {host}:{port}")
    uvicorn.run(
        "serve.api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
