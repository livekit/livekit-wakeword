"""
edge-tts backend for livekit-wakeword.
Integrates Microsoft Edge TTS (edge-tts package) as a SpeechSynthesizer backend.
Produces diverse Spanish (and multilingual) voice samples for wake word training.
"""
from __future__ import annotations

import asyncio
import itertools
import io
import logging
import struct
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Intentamos importar nest_asyncio para compatibilidad con Colab/Jupyter.
# Si no está instalado, lo ignoramos (solo aplica en entornos con event loop activo).
try:
    import nest_asyncio
    nest_asyncio.apply()
    _NEST_ASYNCIO_AVAILABLE = True
except ImportError:
    _NEST_ASYNCIO_AVAILABLE = False


def _run_async(coro):
    """
    Ejecuta una coroutine de forma segura, manejando el caso de Colab
    (donde ya hay un event loop corriendo).
    """
    try:
        loop = asyncio.get_running_loop()
        # Estamos dentro de un loop (Colab/Jupyter). nest_asyncio ya fue aplicado arriba.
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No hay loop corriendo (entorno normal), creamos uno nuevo.
        return asyncio.run(coro)


class EdgeTtsBackend:
    """
    SpeechSynthesizer implementation using Microsoft edge-tts.

    Cicla entre múltiples voces en español para generar diversidad de hablantes,
    similar a como PiperVitsBackend cicla entre speaker pairs con SLERP.

    Requiere:
        pip install edge-tts soundfile numpy

    Para Colab también:
        pip install nest_asyncio
    """

    def __init__(self, config) -> None:
        self._config = config
        self._tts_cfg = config.edge_tts_tts

    def validate_artifacts(self) -> None:
        """
        Verifica que edge-tts esté instalado.
        No requiere archivos locales (usa servidores de Microsoft).
        """
        try:
            import edge_tts  # noqa: F401
        except ImportError as e:
            raise FileNotFoundError(
                "edge-tts no está instalado. "
                "Instálalo con: pip install edge-tts"
            ) from e

        try:
            import soundfile  # noqa: F401
        except ImportError as e:
            raise FileNotFoundError(
                "soundfile no está instalado. "
                "Instálalo con: pip install soundfile"
            ) from e

    def synthesize_clips(
        self,
        phrases: list[str],
        output_dir: Path,
        n_samples: int,
        *,
        start_index: int = 0,
        batch_size: int = 50,
    ) -> None:
        """
        Sintetiza clips de audio y los escribe como clip_%06d.wav a 16kHz.

        Args:
            phrases: Lista de frases a sintetizar (ej: ["Celeste", "hey Celeste"]).
            output_dir: Directorio donde escribir los archivos WAV.
            n_samples: Número total de clips a generar.
            start_index: Índice inicial (para reanudar generación interrumpida).
            batch_size: No usado directamente en edge-tts (se genera de a 1).
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        voices = self._tts_cfg.voices
        rate = self._tts_cfg.rate
        pitch = self._tts_cfg.pitch

        # Construimos un iterador infinito que cicla por:
        # (voice, phrase) — el producto cartesiano, igual que Piper cicla por (speaker_i, speaker_j)
        voice_phrase_cycle = itertools.cycle(
            itertools.product(voices, phrases)
        )

        # Avanzamos el iterador hasta start_index para respetar la reanudación
        for _ in range(start_index):
            next(voice_phrase_cycle)

        generated = 0
        idx = start_index

        while generated < (n_samples - start_index):
            voice, phrase = next(voice_phrase_cycle)
            out_path = output_dir / f"clip_{idx:06d}.wav"

            try:
                audio_int16, sample_rate = _run_async(
                    _synthesize_single(phrase, voice, rate, pitch)
                )

                if audio_int16 is None or len(audio_int16) == 0:
                    logger.warning(
                        "edge-tts devolvió audio vacío para '%s' con voz '%s'. "
                        "Reintentando con siguiente voz.",
                        phrase, voice
                    )
                    # No incrementamos idx ni generated, reintentamos
                    continue

                _write_wav_16k(out_path, audio_int16, sample_rate)
                idx += 1
                generated += 1

                if generated % 50 == 0:
                    logger.info(
                        "edge-tts: generados %d/%d clips en %s",
                        generated, n_samples - start_index, output_dir
                    )

            except Exception as exc:
                logger.warning(
                    "Error sintetizando '%s' con voz '%s': %s. Saltando.",
                    phrase, voice, exc
                )
                # Saltamos este intento pero sí avanzamos para no quedar en loop infinito
                idx += 1
                generated += 1


# ---------------------------------------------------------------------------
# Funciones auxiliares (async y de I/O)
# ---------------------------------------------------------------------------

async def _synthesize_single(
    text: str,
    voice: str,
    rate: str,
    pitch: str,
) -> tuple[np.ndarray, int]:
    """
    Sintetiza una frase con edge-tts y devuelve (audio_int16, sample_rate).

    edge-tts entrega audio MP3 en chunks. Lo decodificamos usando wave
    o soundfile según el formato del subtype.
    """
    import edge_tts
    import soundfile as sf

    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)

    # Recolectamos los bytes de audio del stream
    audio_bytes = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes.extend(chunk["data"])

    if not audio_bytes:
        return np.array([], dtype=np.int16), 16000

    # edge-tts entrega MP3. Lo decodificamos con soundfile + un buffer.
    # soundfile puede leer MP3 si tiene libsndfile con soporte MP3,
    # pero para ser seguros usamos pydub si está disponible, o io.BytesIO directo.
    audio_float, sr = _decode_audio_bytes(bytes(audio_bytes))

    # Resamplear a 16000 Hz si es necesario
    if sr != 16000:
        audio_float = _resample(audio_float, sr, 16000)
        sr = 16000

    # Convertir a mono si es estéreo
    if audio_float.ndim > 1:
        audio_float = audio_float.mean(axis=1)

    # Normalizar y convertir a int16
    max_val = np.max(np.abs(audio_float))
    if max_val > 0:
        audio_float = audio_float / max_val
    audio_int16 = (audio_float * 32767).astype(np.int16)

    return audio_int16, sr


def _decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Decodifica bytes de audio (MP3 desde edge-tts) a float32 numpy array.
    Intenta múltiples métodos según lo que esté disponible.
    """
    # Método 1: soundfile (más rápido, pero requiere soporte MP3 en libsndfile)
    try:
        import soundfile as sf
        import io
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        return audio, sr
    except Exception:
        pass

    # Método 2: pydub (alternativa robusta para MP3)
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.channels == 2:
            samples = samples.reshape(-1, 2)
        samples = samples / (2 ** (seg.sample_width * 8 - 1))
        return samples, seg.frame_rate
    except Exception:
        pass

    # Método 3: librosa (fallback universal)
    try:
        import librosa
        import io
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        return audio.astype(np.float32), sr
    except Exception as e:
        raise RuntimeError(
            f"No se pudo decodificar el audio de edge-tts. "
            f"Instala una de estas librerías: soundfile (con soporte MP3), pydub, o librosa. "
            f"Error original: {e}"
        )


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resamplea audio usando librosa o scipy."""
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        pass

    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(orig_sr, target_sr)
        return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)
    except ImportError:
        raise RuntimeError(
            "Se necesita librosa o scipy para resamplear. "
            "Instala con: pip install librosa"
        )


def _write_wav_16k(path: Path, audio_int16: np.ndarray, sample_rate: int) -> None:
    """Escribe un array int16 como archivo WAV mono a 16kHz."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)          # mono
        wf.setsampwidth(2)          # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())