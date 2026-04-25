"""Default VoxCPM2 model location and voice-design diversification."""

from __future__ import annotations

# Hugging Face repo for ``setup`` snapshot_download.
MODEL_ID = "openbmb/VoxCPM2"
MODEL_CACHE_RELPATH = "voxcpm/VoxCPM2"

# Classifier-free guidance and diffusion steps — multiple defaults for clip diversity.
CFG_VALUES: list[float] = [1.5, 2.0, 2.5, 3.0]
INFERENCE_TIMESTEPS: list[int] = [8, 10, 12]

# Voice design personas (neutral, professional descriptors). Combined with cfg × steps
# for broad synthetic speaker coverage without reference audio.
VOICE_DESIGN_PROMPTS: tuple[str, ...] = (
    "A young adult woman, clear mid-pitch voice, moderate pace, calm and professional",
    "A young adult man, warm baritone, steady pace, friendly and articulate",
    "A middle-aged woman, slightly low pitch, measured pace, confident tone",
    "A middle-aged man, deep resonant voice, slow deliberate pace",
    "An older adult woman, soft gentle voice, slower pace, kind tone",
    "An older adult man, gravelly tenor, moderate pace, matter-of-fact",
    "A young woman, bright energetic tone, slightly faster pace",
    "A young man, light tenor, quick conversational pace",
    "A woman in her thirties, smooth alto, neutral American accent, even pace",
    "A man in his thirties, clear baritone, businesslike pace",
    "A speaker with a higher pitch, enthusiastic and upbeat, medium-fast pace",
    "A speaker with a lower pitch, relaxed and laid-back, slower pace",
    "A young adult, gender-neutral delivery, soft volume, careful enunciation",
    "A confident presenter voice, strong projection, moderate speed",
    "A quiet intimate voice, close-mic feel, slow and clear",
    "A news-anchor style voice, authoritative, even rhythm",
    "A friendly customer-service tone, slightly smiling, medium pace",
    "A tired but clear voice, subdued energy, steady pace",
    "A cheerful animated voice, wide pitch range, lively pace",
    "A serious formal voice, minimal emotion, precise articulation",
    "A Southern US English accent, warm tone, conversational pace",
    "A British English accent, clear RP-like delivery, moderate pace",
    "A speaker with slight vocal fry, casual young adult, medium pace",
    "A very smooth polished voice, studio quality, slow to medium pace",
    "A nasal bright tone, energetic, faster than average pace",
    "A breathy soft voice, gentle, slower pace",
    "A robust athletic-sounding voice, strong and direct, medium pace",
    "A scholarly tone, thoughtful pauses, slower academic pace",
    "A teenager-sounding voice, casual, slightly fast, light pitch",
    "A mature executive voice, controlled low dynamics, steady",
    "A sing-song playful tone, varied pitch, medium pace",
    "A monotone flat delivery, robotic clarity, even speed",
    "A husky voice, low energy, medium-slow pace",
    "A crisp precise voice, minimal accent, fast clear speech",
)


def voice_design_prompts() -> list[str]:
    return list(VOICE_DESIGN_PROMPTS)
