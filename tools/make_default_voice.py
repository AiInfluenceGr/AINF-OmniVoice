"""Generate the built-in demo voice: Aria.

Loads the OmniVoice model, generates a short reference clip, encodes it as a
VoiceClonePrompt, and writes voices/Aria.{pt,json,wav} into assets/default_voice/
so the app can ship them and copy them in on first launch.
"""
import os, sys, json, shutil, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "repo"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import soundfile as sf
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt

OUT = os.path.join(ROOT, "assets", "default_voice")
os.makedirs(OUT, exist_ok=True)

TEXT = "Hello! Welcome to OmniVoice. Let's make some great voices together."
CFG = dict(num_step=32, guidance_scale=2.0, t_shift=0.1, position_temperature=5.0,
           class_temperature=0.0, layer_penalty_factor=5.0, denoise=True,
           preprocess_prompt=True, postprocess_output=True, audio_chunk_duration=15.0,
           audio_chunk_threshold=30.0, pad_duration=0.0, fade_duration=0.0)

print("Loading model ...", flush=True)
t0 = time.time()
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda", dtype=torch.float16)
print(f"Loaded in {time.time()-t0:.1f}s, sr={model.sampling_rate}", flush=True)

print("Generating reference clip ...", flush=True)
audio = model.generate(text=TEXT, generation_config=OmniVoiceGenerationConfig(**CFG))[0]
sr = int(model.sampling_rate)
ref = os.path.join(OUT, "Aria.wav")
sf.write(ref, audio, sr)
print(f"Ref: {len(audio)/sr:.1f}s -> {ref}", flush=True)

print("Encoding voice clone prompt ...", flush=True)
prompt = model.create_voice_clone_prompt(ref_audio=ref, ref_text=TEXT, preprocess_prompt=True)
pt = os.path.join(OUT, "Aria.pt")
prompt.save(pt)

meta = {
    "name": "Aria",
    "ref_text": prompt.ref_text,
    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "ref_audio": "Aria.wav",
    "builtin": True,
}
with open(os.path.join(OUT, "Aria.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("Done. Files:", flush=True)
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print(f"  {f}  {os.path.getsize(p)/1024:.0f} KB")
