"""
Prueba rápida del EdgeTtsBackend sin correr el pipeline completo.
Valida: instalación, síntesis, formato WAV, y sample rate.
"""
import wave
from pathlib import Path
import tempfile

def test_backend_directo():
    """Prueba el backend directamente sin pasar por WakeWordConfig."""
    from livekit.wakeword.data.tts.edge_tts_backend import EdgeTtsBackend

    # Config mínima simulada
    class FakeTtsCfg:
        voices = ["es-PE-AlexNeural", "es-PE-CamilaNeural"]
        rate = "+0%"
        pitch = "+0Hz"

    class FakeConfig:
        edge_tts_tts = FakeTtsCfg()

    backend = EdgeTtsBackend(FakeConfig())

    print("→ Validando artefactos...")
    backend.validate_artifacts()
    print("  ✓ edge-tts y soundfile instalados")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "test_clips"

        print("→ Sintetizando 4 clips de prueba...")
        backend.synthesize_clips(
            phrases=["Celeste", "hey Celeste"],
            output_dir=out_dir,
            n_samples=4,
            start_index=0,
            batch_size=4,
        )

        clips = sorted(out_dir.glob("clip_*.wav"))
        print(f"  ✓ Clips generados: {len(clips)}")
        assert len(clips) == 4, f"Se esperaban 4, se generaron {len(clips)}"

        print("→ Verificando formato de cada clip...")
        for clip in clips:
            with wave.open(str(clip), "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                frames = wf.getnframes()
                duration_ms = frames / sr * 1000

                assert sr == 16000,     f"{clip.name}: sample rate es {sr}, se esperaba 16000"
                assert channels == 1,   f"{clip.name}: debe ser mono, tiene {channels} canales"
                assert sampwidth == 2,  f"{clip.name}: debe ser 16-bit, sampwidth={sampwidth}"
                assert frames > 0,      f"{clip.name}: archivo vacío"

                print(f"  ✓ {clip.name}: {sr}Hz, mono, 16-bit, {duration_ms:.0f}ms")

        print("→ Probando reanudación (start_index=2)...")
        backend.synthesize_clips(
            phrases=["Celeste"],
            output_dir=out_dir,
            n_samples=6,
            start_index=4,
            batch_size=4,
        )
        clips_after = sorted(out_dir.glob("clip_*.wav"))
        assert len(clips_after) == 6, f"Se esperaban 6, hay {len(clips_after)}"
        print(f"  ✓ Reanudación OK: {len(clips_after)} clips en total")

    print("\n✅ Todo OK — el backend está listo para usar en Colab")


def test_config_yaml():
    """Prueba que WakeWordConfig cargue correctamente con tts_backend: edge_tts."""
    from livekit.wakeword import load_config
    import yaml, tempfile, os

    config_data = {
        "model_name": "celeste_test",
        "target_phrases": ["Celeste"],
        "tts_backend": "edge_tts",
        "edge_tts_tts": {
            "voices": ["es-PE-AlexNeural", "es-PE-CamilaNeural"],
            "rate": "+0%",
            "pitch": "+0Hz",
        },
        "n_samples": 100,
        "n_samples_val": 20,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        tmp_path = f.name

    try:
        config = load_config(tmp_path)
        assert config.tts_backend.value == "edge_tts"
        assert config.edge_tts_tts.voices[0] == "es-PE-AlexNeural"
        print("✓ WakeWordConfig carga edge_tts correctamente")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_config_yaml()
    print()
    test_backend_directo()