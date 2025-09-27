import queue
import threading
import time
import re
from typing import List, Optional

import numpy as np
from faster_whisper import WhisperModel
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import streamlit as st

# Keywords that end capture
TERMINATORS = {"done", "enter", "submit", "send"}

@st.cache_resource(show_spinner=False)
def get_whisper_model(model_name: str = "small"):
    # Change to "base", "tiny", etc for speed; or a path to a local model
    return WhisperModel(model_name, device="cpu", compute_type="int8")

def normalize_text(txt: str) -> str:
    return re.sub(r"\s+", " ", txt).strip()

class StreamingTranscriber:
    """
    Accumulates raw PCM audio frames from WebRTC, periodically runs Whisper
    on buffered audio, and exposes partial + final transcripts.
    """
    def __init__(self, terminators=TERMINATORS, chunk_seconds=4.0, sample_rate=16000):
        self.terminators = terminators
        self.chunk_seconds = chunk_seconds
        self.sample_rate = sample_rate

        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.buffer: List[np.ndarray] = []
        self.partial_text = ""
        self.final_text = ""
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._finished = False

    def start(self, model_name="small"):
        if self._thread and self._thread.is_alive():
            return
        self.model = get_whisper_model(model_name)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        # Final transcription on leftovers
        if self.buffer:
            self._transcribe_buffer(final=True)
        self._finished = True

    def add_audio_frames(self, frames: List[np.ndarray]):
        for f in frames:
            self.audio_q.put(f)

    def is_finished(self):
        return self._finished

    def _loop(self):
        last_transcribe = time.time()
        min_samples = int(self.chunk_seconds * self.sample_rate)
        collected = 0
        temp_accum: List[np.ndarray] = []

        while not self._stop_event.is_set():
            try:
                frame = self.audio_q.get(timeout=0.3)
                temp_accum.append(frame)
                collected += len(frame)
            except queue.Empty:
                pass

            # Periodically move accumulated chunk to buffer and transcribe
            if collected >= min_samples:
                self.buffer.append(np.concatenate(temp_accum))
                temp_accum.clear()
                collected = 0
                self._transcribe_buffer()
                if self._check_termination():
                    self.stop()
                    break

            # Safety to avoid tight loop
            if (time.time() - last_transcribe) > 15 and temp_accum:
                # Force a smaller transcription if user pauses
                self.buffer.append(np.concatenate(temp_accum))
                temp_accum.clear()
                collected = 0
                self._transcribe_buffer()
                if self._check_termination():
                    self.stop()
                    break
                last_transcribe = time.time()

        # Drain remainder if stopping
        if temp_accum:
            self.buffer.append(np.concatenate(temp_accum))
            self._transcribe_buffer(final=True)
            self._check_termination()

    def _transcribe_buffer(self, final=False):
        if not self.buffer:
            return
        # Concatenate all buffered raw mono audio (float32 PCM)
        full = np.concatenate(self.buffer)
        self.buffer.clear()
        # Convert float32 PCM (-1..1) to int16 for Whisper (if needed)
        # faster-whisper accepts numpy float32 waveform at 16k too; resample if not 16k
        # We assume frames are already 16k mono.
        segments, _info = self.model.transcribe(full, beam_size=1, vad_filter=True)
        new_text_parts = []
        for seg in segments:
            new_text_parts.append(seg.text)
        appended = normalize_text(" ".join(new_text_parts))
        if appended:
            if self.partial_text:
                self.partial_text += " " + appended
            else:
                self.partial_text = appended
        if final:
            self.final_text = self.partial_text

    def _check_termination(self) -> bool:
        # Check last word in partial transcript
        words = self.partial_text.lower().split()
        if not words:
            return False
        last_word = re.sub(r"[^\w]", "", words[-1])
        if last_word in self.terminators:
            # Remove the terminator keyword from final text
            cleaned = re.sub(rf"\b({ '|'.join(self.terminators) })\b\.?$", "", self.partial_text, flags=re.IGNORECASE)
            self.final_text = normalize_text(cleaned)
            return True
        return False


def voice_capture_ui(key="voice_capture", model_name="small"):
    """
    High-level UI wrapper. Returns (final_text, is_listening, partial_text).
    final_text is set once a terminator word is detected or user stops manually.
    """
    if "voice_state" not in st.session_state:
        st.session_state.voice_state = {
            "listening": False,
            "transcriber": None,
            "partial": "",
            "final": "",
        }

    state = st.session_state.voice_state

    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        if not state["listening"]:
            if st.button("🎤 Start Voice", key=f"{key}_start"):
                state["transcriber"] = StreamingTranscriber()
                state["transcriber"].start(model_name=model_name)
                state["listening"] = True
        else:
            if st.button("🛑 Stop", key=f"{key}_stop"):
                state["transcriber"].stop()
                state["listening"] = False

    with col2:
        st.write("Status:", "Listening..." if state["listening"] else "Idle")

    # WebRTC streamer (audio only)
    webrtc_ctx = webrtc_streamer(
        key=f"{key}_webrtc",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"audio": True, "video": False},
        audio_receiver_size=256,
        async_transform=False,
    )

    # Collect audio frames
    if state["listening"] and webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.audio_receiver:
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=0.1)
            np_frames = []
            for f in audio_frames:
                # f is av.AudioFrame
                arr = f.to_ndarray()
                # Convert shape (channels, samples) -> mono
                if arr.ndim == 2:
                    arr = np.mean(arr, axis=0)
                # Normalize to float32 -1..1
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32) / 32768.0 if arr.dtype == np.int16 else arr.astype(np.float32)
                np_frames.append(arr)
            if np_frames and state["transcriber"]:
                state["transcriber"].add_audio_frames(np_frames)
        except Exception:
            pass

    # Update partial / final
    if state["transcriber"]:
        state["partial"] = state["transcriber"].partial_text
        if state["transcriber"].final_text and not state["transcriber"].is_finished():
            # will be set only when termination word recognized; stop automatically
            state["transcriber"].stop()
            state["listening"] = False
        if state["transcriber"].is_finished():
            state["final"] = state["transcriber"].final_text

    if state["partial"] and not state["final"]:
        st.info(f"Partial: {state['partial']}")
    if state["final"]:
        st.success(f"Final: {state['final']}")

    return state["final"], state["listening"], state["partial"]