"""
OmniVoice control panel (AIINFLUENCE edition).

Gradio UI for k2-fsa/OmniVoice: voice cloning, voice design, auto voice,
batch generation, saved voices, output history and model management.

Run:  .venv\\Scripts\\python.exe app.py  [--port 7870] [--listen] [--preload]
"""

import argparse
import base64
import collections
import datetime as dt
import gc
import json
import logging
import os

# Gradio phones home for usage analytics by default; this app stays local.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import re
import shutil
import subprocess
import threading
import time
import zipfile

import gradio as gr
import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(ROOT, "voices")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
FONTS_DIR = os.path.join(ROOT, "assets", "fonts")
SETTINGS_FILE = os.path.join(ROOT, "settings.json")
for _d in (VOICES_DIR, OUTPUTS_DIR):
    os.makedirs(_d, exist_ok=True)

DEFAULT_SETTINGS = {
    "model": "k2-fsa/OmniVoice",
    "device": "cuda",
    "dtype": "float16",
    "asr_model": "openai/whisper-large-v3-turbo",
    "lora_adapter": "",
}

NONVERBAL_TAGS = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]",
    "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]",
    "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]",
    "[dissatisfaction-hnn]",
]

DESIGN_CATEGORIES = {
    "Gender": ["male", "female"],
    "Age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
    "Pitch": ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
    "Style": ["whisper"],
    "English accent": [
        "american accent", "british accent", "australian accent", "canadian accent",
        "indian accent", "chinese accent", "korean accent", "japanese accent",
        "portuguese accent", "russian accent",
    ],
    "Chinese dialect": [
        "河南话", "陕西话", "四川话", "贵州话", "云南话", "桂林话",
        "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话",
    ],
}
DESIGN_INFO = {
    "English accent": "Only applies when the text is English.",
    "Chinese dialect": "Only applies when the text is Chinese.",
    "Style": "Whisper is the only style.",
}
ANY = "any"

# ---------------------------------------------------------------------------
# Language list (read from the package without importing torch-heavy modules)
# ---------------------------------------------------------------------------
try:
    from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

    LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)
except Exception:  # pragma: no cover
    LANGUAGES = ["Auto", "English", "Chinese"]

# ---------------------------------------------------------------------------
# Console log (ring buffer shared by every tab)
# ---------------------------------------------------------------------------
LOG = collections.deque(maxlen=400)


def log(msg: str) -> None:
    line = f"{dt.datetime.now():%H:%M:%S}  {msg}"
    LOG.append(line)
    print(line, flush=True)


class _BufHandler(logging.Handler):
    def emit(self, record):
        try:
            if record.levelno >= logging.INFO and record.name.startswith("omnivoice"):
                log(record.getMessage())
        except Exception:
            pass


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logging.getLogger().addHandler(_BufHandler())


def console_text() -> str:
    return "\n".join(LOG) if LOG else "Ready. Nothing has run yet."


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def load_settings() -> dict:
    s = dict(DEFAULT_SETTINGS)
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                s.update(json.load(f))
        except Exception as e:
            log(f"settings.json unreadable, using defaults ({e})")
    return s


def save_settings(s: dict) -> None:
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Model holder (one model in memory, one generation at a time)
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self):
        self.model = None
        self.info = {}
        self.lock = threading.Lock()
        self.busy = ""

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self, model_id, device, dtype, asr_model, lora, load_asr=False):
        from omnivoice import OmniVoice

        self.unload(quiet=True)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this Python environment. Pick cpu.")
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        if device == "cpu" and torch_dtype == torch.float16:
            torch_dtype = torch.float32
            log("fp16 is not supported on CPU, using float32 instead.")
        log(f"Loading {model_id} on {device} ({str(torch_dtype).replace('torch.', '')}) ...")
        t0 = time.time()
        model = OmniVoice.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch_dtype,
            load_asr=load_asr,
            asr_model_name=asr_model or None,
        )
        if lora:
            from omnivoice.utils.lora import load_lora_adapter

            log(f"Applying LoRA adapter {lora} ...")
            model = load_lora_adapter(model, lora)
        self.model = model
        self.info = {
            "model": model_id, "device": device, "dtype": str(torch_dtype).replace("torch.", ""),
            "lora": lora or "", "sr": int(model.sampling_rate),
        }
        log(f"Model ready in {time.time() - t0:.1f}s. Sample rate {model.sampling_rate} Hz.")

    def unload(self, quiet=False):
        if self.model is None:
            if not quiet:
                log("No model in memory.")
            return
        self.model = None
        self.info = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not quiet:
            log("Model unloaded, VRAM released.")

    def ensure(self):
        if self.model is None:
            s = load_settings()
            self.load(s["model"], s["device"], s["dtype"], s["asr_model"], s["lora_adapter"])
        return self.model


ENGINE = Engine()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NAME_RE = re.compile(r"^[A-Za-z0-9 _\-\.]{1,60}$")


def gpu_stats():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip().splitlines()[0]
        used, total, name = [x.strip() for x in out.split(",", 2)]
        return int(used), int(total), name
    except Exception:
        return None


def list_voices():
    names = []
    for f in sorted(os.listdir(VOICES_DIR)):
        if f.endswith(".pt"):
            names.append(f[:-3])
    return names


def voice_meta(name):
    p = os.path.join(VOICES_DIR, name + ".json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def free_voice_name(name):
    base = re.sub(r"_v\d+$", "", name)
    n = 2
    while os.path.exists(os.path.join(VOICES_DIR, f"{base}_v{n}.pt")):
        n += 1
    return f"{base}_v{n}"


def slug(text, n=32):
    s = re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-").lower()
    return (s[:n].strip("-")) or "clip"


def build_config(steps, guidance, t_shift, pos_temp, cls_temp, layer_pen, denoise,
                 pre, post, chunk_dur, chunk_thr, pad, fade):
    from omnivoice import OmniVoiceGenerationConfig

    return OmniVoiceGenerationConfig(
        num_step=int(round(steps or 32)),
        guidance_scale=float(2.0 if guidance is None else guidance),
        t_shift=float(0.1 if t_shift is None else t_shift),
        position_temperature=float(5.0 if pos_temp is None else pos_temp),
        class_temperature=float(0.0 if cls_temp is None else cls_temp),
        layer_penalty_factor=float(5.0 if layer_pen is None else layer_pen),
        denoise=bool(denoise),
        preprocess_prompt=bool(pre),
        postprocess_output=bool(post),
        audio_chunk_duration=float(chunk_dur or 15.0),
        audio_chunk_threshold=float(chunk_thr or 30.0),
        pad_duration=float(0.0 if pad is None else pad),
        fade_duration=float(0.0 if fade is None else fade),
    )


def apply_seed(seed):
    seed = int(seed) if seed is not None and int(seed) >= 0 else int(np.random.randint(0, 2**31 - 1))
    from omnivoice.utils.common import fix_random_seed

    fix_random_seed(seed)
    return seed


def save_output(audio, sr, mode, text, params):
    day = dt.datetime.now().strftime("%Y-%m-%d")
    folder = os.path.join(OUTPUTS_DIR, day)
    os.makedirs(folder, exist_ok=True)
    stem = f"{dt.datetime.now():%H%M%S}_{mode}_{slug(text)}"
    path = os.path.join(folder, stem + ".wav")
    k = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem}_{k}.wav")
        k += 1
    sf.write(path, audio, sr)
    meta = dict(params, mode=mode, text=text, file=os.path.basename(path),
                seconds=round(len(audio) / sr, 2), created=dt.datetime.now().isoformat(timespec="seconds"))
    with open(path[:-4] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------
def run_generate(mode, text, *, language, settings, instruct=None, ref_audio=None,
                 ref_text=None, voice_name=None, save_as=None):
    """Returns (audio_path or None, status message)."""
    text = (text or "").strip()
    if not text:
        return None, "Type the text you want spoken first."
    if not ENGINE.lock.acquire(blocking=False):
        return None, f"Busy: {ENGINE.busy or 'another job'} is still running. Wait for it to finish."
    try:
        ENGINE.busy = f"{mode} generation"
        model = ENGINE.ensure()
        (steps, guidance, t_shift, pos_temp, cls_temp, layer_pen, denoise, pre, post,
         chunk_dur, chunk_thr, pad, fade, speed, duration, normalize, seed) = settings
        cfg = build_config(steps, guidance, t_shift, pos_temp, cls_temp, layer_pen, denoise,
                           pre, post, chunk_dur, chunk_thr, pad, fade)
        if normalize:
            new = normalize_numbers(text, language)
            if new != text:
                log(f"Normalized text: {new}")
            text = new
        kw = dict(text=text, generation_config=cfg)
        lang = language if language and language != "Auto" else None
        kw["language"] = lang
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)
        elif speed is not None and abs(float(speed) - 1.0) > 1e-6:
            kw["speed"] = float(speed)
        if instruct:
            kw["instruct"] = instruct

        params = {"language": language, "instruct": instruct or "", "config": cfg.__dict__,
                  "speed": kw.get("speed", 1.0), "duration": kw.get("duration"),
                  "normalize_text": bool(normalize)}

        if mode == "clone":
            from omnivoice import VoiceClonePrompt

            if voice_name:
                pt = os.path.join(VOICES_DIR, voice_name + ".pt")
                if not os.path.exists(pt):
                    return None, f"Saved voice '{voice_name}' not found. Refresh the list."
                prompt = VoiceClonePrompt.load(pt)
                params["voice"] = voice_name
                log(f"Using saved voice '{voice_name}'.")
            else:
                if not ref_audio:
                    return None, "Upload or record a reference clip, or pick a saved voice."
                rt = (ref_text or "").strip() or None
                if rt is None:
                    log("No reference transcript given, transcribing with Whisper ...")
                log("Encoding reference audio ...")
                prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=rt,
                                                         preprocess_prompt=bool(pre))
                params["ref_text"] = prompt.ref_text
                if save_as:
                    msg = save_voice(prompt, save_as, ref_audio)
                    log(msg)
            kw["voice_clone_prompt"] = prompt

        used_seed = apply_seed(seed)
        params["seed"] = used_seed
        log(f"Generating ({mode}, {cfg.num_step} steps, seed {used_seed}) ...")
        t0 = time.time()
        audio = model.generate(**kw)[0]
        el = time.time() - t0
        sr = int(model.sampling_rate)
        secs = len(audio) / sr
        path = save_output(audio, sr, mode, text, params)
        rtf = el / secs if secs else 0
        log(f"Done: {secs:.1f}s of audio in {el:.1f}s (RTF {rtf:.3f}). Saved {os.path.relpath(path, ROOT)}")
        return path, f"{secs:.1f}s of audio in {el:.1f}s · seed {used_seed} · {os.path.relpath(path, ROOT)}"
    except Exception as e:
        logging.exception("generation failed")
        log(f"Error: {type(e).__name__}: {e}")
        return None, f"Error: {type(e).__name__}: {e}"
    finally:
        ENGINE.busy = ""
        ENGINE.lock.release()


def save_voice(prompt, name, ref_audio=None):
    name = (name or "").strip()
    if not NAME_RE.match(name):
        return "Voice not saved: use letters, numbers, spaces, - _ . (max 60)."
    pt = os.path.join(VOICES_DIR, name + ".pt")
    if os.path.exists(pt):
        return f"Voice not saved: '{name}' already exists. Try '{free_voice_name(name)}'."
    prompt.save(pt)
    meta = {"name": name, "ref_text": prompt.ref_text, "created": dt.datetime.now().isoformat(timespec="seconds")}
    if ref_audio and os.path.exists(ref_audio):
        ext = os.path.splitext(ref_audio)[1] or ".wav"
        shutil.copyfile(ref_audio, os.path.join(VOICES_DIR, name + ext))
        meta["ref_audio"] = name + ext
    with open(os.path.join(VOICES_DIR, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return f"Saved voice '{name}'."


def normalize_numbers(text, language):
    """Spell out numbers. Uses OmniVoice's WeTextProcessing path when it is
    installed; otherwise falls back to num2words (WeTextProcessing needs pynini,
    which has no Windows wheel)."""
    try:
        from omnivoice.utils.text import normalize_text

        return normalize_text(text, None if language in (None, "Auto") else language)
    except ImportError:
        pass
    from omnivoice.utils.text import _apply_with_protection, _num2words_segment, _resolve_lang_code

    code = _resolve_lang_code(None if language in (None, "Auto") else language, text)
    return _apply_with_protection(text, lambda seg: _num2words_segment(seg, code), protect_pinyin=True)


def design_instruct(free_text, *picks):
    parts = [p for p in picks if p and p != ANY]
    extra = (free_text or "").strip()
    if extra:
        parts += [x.strip() for x in re.split(r"[,，]", extra) if x.strip()]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def h_clone(text, lang, source, voice, ref_audio, ref_text, save_as, instruct, *settings):
    use_saved = source == "Saved voice"
    if use_saved and not voice:
        return None, "Pick a saved voice, or switch the source to a new reference clip."
    return run_generate(
        "clone", text, language=lang, settings=settings,
        instruct=(instruct or "").strip() or None,
        ref_audio=None if use_saved else ref_audio,
        ref_text=ref_text, voice_name=voice if use_saved else None,
        save_as=None if use_saved else ((save_as or "").strip() or None),
    )


def h_design(text, lang, free, *rest):
    picks = rest[: len(DESIGN_CATEGORIES)]
    settings = rest[len(DESIGN_CATEGORIES):]
    instr = design_instruct(free, *picks)
    if not instr:
        return None, "Pick at least one attribute or type a description. For a random voice use 03 · AUTO."
    log(f"Voice design: {instr}")
    return run_generate("design", text, language=lang, settings=settings, instruct=instr)


def h_auto(text, lang, *settings):
    return run_generate("auto", text, language=lang, settings=settings)


def h_transcribe(ref_audio):
    if not ref_audio:
        return gr.update(), "Upload a reference clip first."
    if not ENGINE.lock.acquire(blocking=False):
        return gr.update(), f"Busy: {ENGINE.busy or 'another job'} is running."
    try:
        ENGINE.busy = "transcription"
        model = ENGINE.ensure()
        if model._asr_pipe is None:
            log("Loading Whisper for transcription (first time downloads ~1.6 GB) ...")
            model.load_asr_model()
        from omnivoice.utils.audio import load_audio

        wav = load_audio(ref_audio, model.sampling_rate)
        txt = model.transcribe((wav, model.sampling_rate))
        log(f"Transcript: {txt}")
        return txt, "Transcript filled in. Fix any mistakes before generating."
    except Exception as e:
        log(f"Error: {type(e).__name__}: {e}")
        return gr.update(), f"Error: {type(e).__name__}: {e}"
    finally:
        ENGINE.busy = ""
        ENGINE.lock.release()


def h_voice_info(name):
    if not name:
        return None, ""
    m = voice_meta(name)
    ra = m.get("ref_audio")
    path = os.path.join(VOICES_DIR, ra) if ra else None
    return (path if path and os.path.exists(path) else None), (m.get("ref_text") or "")


def h_refresh_voices(current=None):
    v = list_voices()
    return gr.update(choices=v, value=current if current in v else (v[0] if v else None))


def h_batch(lines, lang, source, voice, free, *rest):
    picks = rest[: len(DESIGN_CATEGORIES)]
    rest = rest[len(DESIGN_CATEGORIES):]
    batch_size = int(rest[0] or 4)
    settings = rest[1:]
    texts = [ln.strip() for ln in (lines or "").splitlines() if ln.strip()]
    if not texts:
        return None, None, "Paste at least one line of text. Each line becomes one clip."
    if not ENGINE.lock.acquire(blocking=False):
        return None, None, f"Busy: {ENGINE.busy or 'another job'} is running."
    try:
        ENGINE.busy = "batch"
        model = ENGINE.ensure()
        (steps, guidance, t_shift, pos_temp, cls_temp, layer_pen, denoise, pre, post,
         chunk_dur, chunk_thr, pad, fade, speed, duration, normalize, seed) = settings
        cfg = build_config(steps, guidance, t_shift, pos_temp, cls_temp, layer_pen, denoise,
                           pre, post, chunk_dur, chunk_thr, pad, fade)
        common = dict(generation_config=cfg)
        if normalize:
            texts = [normalize_numbers(t, lang) for t in texts]
        lng = lang if lang and lang != "Auto" else None
        if speed is not None and abs(float(speed) - 1.0) > 1e-6:
            common["speed"] = float(speed)
        prompt, instr = None, None
        if source == "Saved voice":
            if not voice:
                return None, None, "Pick a saved voice for the batch."
            from omnivoice import VoiceClonePrompt

            prompt = VoiceClonePrompt.load(os.path.join(VOICES_DIR, voice + ".pt"))
        elif source == "Designed voice":
            instr = design_instruct(free, *picks)
            if not instr:
                return None, None, "Pick at least one design attribute for the batch."
        used_seed = apply_seed(seed)
        stamp = dt.datetime.now().strftime("%H%M%S")
        folder = os.path.join(OUTPUTS_DIR, dt.datetime.now().strftime("%Y-%m-%d"), f"batch_{stamp}")
        os.makedirs(folder, exist_ok=True)
        sr = int(model.sampling_rate)
        files = []
        t0 = time.time()
        log(f"Batch: {len(texts)} lines, batch size {batch_size}, seed {used_seed}.")
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            kw = dict(common, text=chunk, language=[lng] * len(chunk) if lng else None)
            if prompt is not None:
                kw["voice_clone_prompt"] = [prompt] * len(chunk)
            if instr:
                kw["instruct"] = [instr] * len(chunk)
            audios = model.generate(**kw)
            for j, a in enumerate(audios):
                idx = i + j + 1
                p = os.path.join(folder, f"{idx:03d}_{slug(chunk[j], 24)}.wav")
                sf.write(p, a, sr)
                files.append(p)
            log(f"Batch: {min(i + batch_size, len(texts))}/{len(texts)} done ({time.time() - t0:.1f}s).")
        with open(os.path.join(folder, "batch.json"), "w", encoding="utf-8") as f:
            json.dump({"lines": texts, "source": source, "voice": voice, "instruct": instr,
                       "language": lang, "seed": used_seed, "config": cfg.__dict__}, f,
                      indent=2, ensure_ascii=False)
        zpath = folder + ".zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, os.path.basename(p))
        msg = f"{len(files)} clips in {time.time() - t0:.1f}s · {os.path.relpath(folder, ROOT)}"
        log("Batch done: " + msg)
        return files[0] if files else None, zpath, msg
    except Exception as e:
        logging.exception("batch failed")
        log(f"Error: {type(e).__name__}: {e}")
        return None, None, f"Error: {type(e).__name__}: {e}"
    finally:
        ENGINE.busy = ""
        ENGINE.lock.release()


def history_choices():
    items = []
    for day in sorted(os.listdir(OUTPUTS_DIR), reverse=True):
        dpath = os.path.join(OUTPUTS_DIR, day)
        if not os.path.isdir(dpath):
            continue
        for f in sorted(os.listdir(dpath), reverse=True):
            if f.endswith(".wav"):
                items.append((f"{day}  {f[:-4]}", os.path.join(dpath, f)))
        if len(items) > 300:
            break
    return items


def h_history_refresh():
    ch = history_choices()
    return gr.update(choices=ch, value=ch[0][1] if ch else None)


def h_history_pick(path):
    if not path or not os.path.exists(path):
        return None, ""
    meta = path[:-4] + ".json"
    info = ""
    if os.path.exists(meta):
        with open(meta, encoding="utf-8") as f:
            m = json.load(f)
        info = json.dumps(m, indent=2, ensure_ascii=False)
    return path, info


def h_open_outputs():
    try:
        os.startfile(OUTPUTS_DIR)  # type: ignore[attr-defined]
        return "Opened the outputs folder."
    except Exception as e:
        return f"Could not open the folder: {e}"


def h_load_model(model_id, device, dtype, asr_model, lora, preload_asr):
    model_id = (model_id or "").strip() or DEFAULT_SETTINGS["model"]
    lora = (lora or "").strip()
    asr_model = (asr_model or "").strip() or DEFAULT_SETTINGS["asr_model"]
    if lora and not os.path.isdir(lora):
        return f"LoRA folder not found: {lora}", header_html()
    if not ENGINE.lock.acquire(blocking=False):
        return f"Busy: {ENGINE.busy or 'another job'} is running.", header_html()
    try:
        ENGINE.busy = "model load"
        s = load_settings()
        s.update(model=model_id, device=device, dtype=dtype, asr_model=asr_model, lora_adapter=lora)
        save_settings(s)
        ENGINE.load(model_id, device, dtype, asr_model, lora, load_asr=bool(preload_asr))
        return "Model loaded. Settings saved.", header_html()
    except Exception as e:
        logging.exception("load failed")
        log(f"Error: {type(e).__name__}: {e}")
        return f"Error: {type(e).__name__}: {e}", header_html()
    finally:
        ENGINE.busy = ""
        ENGINE.lock.release()


def h_unload():
    if not ENGINE.lock.acquire(blocking=False):
        return f"Busy: {ENGINE.busy or 'another job'} is running.", header_html()
    try:
        ENGINE.unload()
        return "Model unloaded.", header_html()
    finally:
        ENGINE.lock.release()


def append_tag(text, tag):
    if not tag or tag == TAG_PICK:
        return gr.update(), gr.update()
    t = (text or "").rstrip()
    return (t + " " + tag).strip() + " ", gr.update(value=TAG_PICK)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def header_html():
    g = gpu_stats()
    if ENGINE.busy:
        model_tag = f'<span class="tag tag-on">{ENGINE.busy.upper()}</span>'
    elif ENGINE.loaded:
        model_tag = '<span class="tag tag-on">MODEL READY</span>'
    else:
        model_tag = '<span class="tag">MODEL NOT LOADED</span>'
    chips = [model_tag]
    if ENGINE.loaded:
        i = ENGINE.info
        chips.append(f'<span class="tag">{i["device"].upper()} · {i["dtype"].upper()}</span>')
        if ENGINE.model is not None and getattr(ENGINE.model, "_asr_pipe", None) is not None:
            chips.append('<span class="tag tag-on">WHISPER</span>')
        if i.get("lora"):
            chips.append('<span class="tag tag-on">LORA</span>')
    chips.append(f'<span class="tag">VOICES {len(list_voices())}</span>')
    if g:
        used, total, name = g
        cls = "tag tag-warn" if (not ENGINE.loaded and used > 0.6 * total) else "tag"
        chips.append(f'<span class="{cls}">GPU {used / 1024:.1f} / {total / 1024:.1f} GB</span>')
    return (
        '<div class="ov-head">'
        '<div class="ov-brand"><span class="ov-mark">AIINFLUENCE</span>'
        '<span class="ov-rule"></span><span class="ov-sub">VOICE LAB</span></div>'
        '<h1 class="ov-title">OmniVoice</h1>'
        '<p class="ov-lede">Zero-shot speech in 600+ languages. Clone a voice from a few seconds of audio, '
        'describe one from attributes, or let the model pick.</p>'
        f'<div class="ov-chips">{"".join(chips)}</div></div>'
    )


def eyebrow(idx, label, text):
    return (
        f'<div class="eyebrow"><span class="eyebrow-index">{idx}</span>'
        f'<span class="eyebrow-rule"></span><span class="eyebrow-label">{label}</span></div>'
        f'<p class="stage-copy">{text}</p>'
    )


# ---------------------------------------------------------------------------
# Theme / CSS
# ---------------------------------------------------------------------------
def font_faces():
    out = []
    spec = [
        ("Inter Tight", "inter-tight-latin-400-normal.woff2", 400, None),
        ("Inter Tight", "inter-tight-latin-500-normal.woff2", 500, None),
        ("Inter Tight", "inter-tight-greek-400-normal.woff2", 400, "U+0370-03FF,U+1F00-1FFF"),
        ("Inter Tight", "inter-tight-greek-500-normal.woff2", 500, "U+0370-03FF,U+1F00-1FFF"),
        ("JetBrains Mono", "jetbrains-mono-latin-400-normal.woff2", 400, None),
        ("JetBrains Mono", "jetbrains-mono-latin-500-normal.woff2", 500, None),
    ]
    for fam, fn, w, rng in spec:
        p = os.path.join(FONTS_DIR, fn)
        if not os.path.exists(p):
            continue
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        r = f"unicode-range:{rng};" if rng else ""
        out.append(f"@font-face{{font-family:'{fam}';font-weight:{w};font-style:normal;"
                   f"font-display:swap;{r}src:url(data:font/woff2;base64,{b64}) format('woff2');}}")
    return "\n".join(out)


C = dict(body="#060906", panel="#0d120e", raised="#131a14", well="#030503", border="#233026",
         text="#c9d6c6", bone="#e6efe1", ash="#a3aea4", steel="#7f8c81", fog="#5d6a60",
         green="#5be35a", green_hi="#8dff8c", teal="#103a18", teal_hi="#17501f",
         warn="#e3b341", fault="#c4566b")

_CSS_TEMPLATE = """
:root, .dark {
  --ov-body:%(body)s; --ov-panel:%(panel)s; --ov-raised:%(raised)s; --ov-well:%(well)s;
  --ov-border:%(border)s; --ov-text:%(text)s; --ov-bone:%(bone)s; --ov-ash:%(ash)s;
  --ov-steel:%(steel)s; --ov-fog:%(fog)s; --ov-green:%(green)s; --ov-green-hi:%(green_hi)s;
  --ov-teal:%(teal)s; --ov-teal-hi:%(teal_hi)s; --ov-warn:%(warn)s; --ov-fault:%(fault)s;
  --ov-mono:'JetBrains Mono', ui-monospace, Consolas, monospace;
  --ov-sans:'Inter Tight', system-ui, sans-serif;
}
html { background: var(--ov-body) !important; }
body, gradio-app, .main, .app, .wrap.app, .contain { background: transparent !important; }
gradio-app { position: relative; z-index: 1; }

/* live board background (assets/board.js): photo + veil + electricity canvas */
.board-bg { position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none; user-select: none;
  background: var(--ov-body); }
.board-img, .board-veil, .board-electric { position: absolute; inset: 0; width: 100%; height: 100%; }
.board-img { object-fit: cover; object-position: 50% 50%; filter: brightness(0.42) saturate(0.95) contrast(1.08); }
.board-veil { background:
  linear-gradient(rgb(4 10 5 / 0.78) 0, rgb(4 10 5 / 0.5) 200px, rgb(4 10 5 / 0) 380px),
  radial-gradient(ellipse 125% 100% at 50% 40%, transparent 40%, rgb(4 10 5 / 0.72) 100%),
  linear-gradient(transparent 50%, rgb(4 10 5 / 0.45) 78%, rgb(4 10 5 / 0.7) 100%); }
.ov-fx-off .board-img { filter: brightness(0.34) saturate(0.9); }
.ov-fx { all: unset; cursor: pointer; font-family: var(--ov-mono); font-size: 11.5px; letter-spacing: .08em;
  color: var(--ov-steel); border: 1px solid var(--ov-border); background: var(--ov-panel); padding: 3px 8px;
  border-radius: 2px; margin-left: 10px; }
.ov-fx::before { content: "["; color: var(--ov-fog); margin-right: 4px; }
.ov-fx::after { content: "]"; color: var(--ov-fog); margin-left: 4px; }
.ov-fx.is-on { color: var(--ov-green); border-color: #1e4a22; }
.ov-fx:hover { color: var(--ov-bone); }
.gradio-container {
  width: min(1480px, 96vw) !important; max-width: none !important; flex-grow: 0 !important;
  margin-left: auto !important; margin-right: auto !important;
  background: transparent !important; color: var(--ov-text) !important;
  font-family: var(--ov-sans) !important;
  --color-accent: var(--ov-green) !important; --color-accent-soft: var(--ov-teal) !important;
  --radius-sm: 2px !important; --radius-md: 2px !important; --radius-lg: 2px !important;
  --radius-xl: 2px !important; --radius-xxl: 2px !important;
  --shadow-drop: none !important; --shadow-drop-lg: none !important; --block-shadow: none !important;
}
.gradio-container * { box-shadow: none !important; }
footer { display: none !important; }

/* header */
.ov-head { padding: 34px 0 22px; border-bottom: 1px solid var(--ov-border); margin-bottom: 4px; }
.ov-title, .ov-lede { text-shadow: 0 1px 12px rgb(3 5 3 / .9); }
.ov-brand { display:flex; align-items:center; gap:12px; font-family:var(--ov-mono); font-size:12px;
  letter-spacing:.16em; color:var(--ov-steel); }
.ov-mark { color: var(--ov-green); }
.ov-rule { width:48px; height:1px; background:var(--ov-border); }
.ov-title { font-family:var(--ov-sans) !important; font-weight:400 !important; font-size:44px !important;
  letter-spacing:-.02em; color:var(--ov-bone) !important; margin:14px 0 6px !important; line-height:1.05; }
.ov-lede { color: var(--ov-ash); margin: 0 0 16px; max-width: 760px; font-size: 15px; }
.ov-chips { display:flex; flex-wrap:wrap; gap:8px; justify-content:flex-start; }
.tag { font-family:var(--ov-mono); font-size:11.5px; letter-spacing:.08em; color:var(--ov-steel);
  border:1px solid var(--ov-border); background:var(--ov-panel); padding:4px 8px; border-radius:2px; }
.tag::before { content:"["; color:var(--ov-fog); margin-right:4px; }
.tag::after { content:"]"; color:var(--ov-fog); margin-left:4px; }
.tag-on { color: var(--ov-green); border-color: #1e4a22; }
.tag-warn { color: var(--ov-warn); border-color: #4a3d1e; }

/* eyebrows */
.eyebrow { display:flex; align-items:center; gap:12px; font-family:var(--ov-mono); font-size:12px;
  letter-spacing:.14em; text-transform:uppercase; margin: 2px 0 8px; }
.eyebrow-index { color: var(--ov-green); }
.eyebrow-rule { width: 40px; height: 1px; background: var(--ov-border); }
.eyebrow-label { color: var(--ov-steel); }
.stage-copy { color: var(--ov-ash) !important; font-size: 14.5px; margin: 0 0 6px; max-width: 900px; }

/* tabs */
.tab-wrapper, [role=tablist] { border-bottom: 1px solid var(--ov-border) !important; }
[role=tab] { font-family: var(--ov-mono) !important; font-size: 12.5px !important; letter-spacing: .1em;
  color: var(--ov-steel) !important; text-transform: uppercase; border: none !important;
  background: transparent !important; padding: 10px 16px !important; }
[role=tab]:hover { color: var(--ov-text) !important; }
[role=tab].selected { color: var(--ov-green) !important; border-bottom: 1px solid var(--ov-green) !important; }
[role=tab].selected::after { background: transparent !important; height: 0 !important; }
.tabitem { border: 1px solid var(--ov-border) !important; border-top: none !important;
  background: rgb(13 18 14 / 0.94) !important; padding: 22px 24px !important; min-height: 1120px; }
#tab-guide .ov-guide { max-height: 960px; overflow-y: auto; padding-right: 14px; }
.tabitem .row, .tabitem .column, .tabitem .form { background: transparent !important; border: none !important; }
.gr-group, .gr-group > .styler, .styler { background: transparent !important; gap: 12px !important; border: none !important; }
.tab-wrapper { border-bottom: none !important; background: rgb(6 9 6 / 0.94) !important; }

/* blocks */
.block, .form { background: transparent !important; border-color: var(--ov-border) !important; }
.block { border-radius: 2px !important; }
label span, .block-title, .block-label, [data-testid=block-label], [data-testid=block-info] {
  font-family: var(--ov-mono) !important; font-size: 11.5px !important; letter-spacing: .1em !important;
  text-transform: uppercase; color: var(--ov-ash) !important; background: transparent !important;
  border: none !important; }
.block-info, [data-testid=block-info], .info { text-transform:none !important; letter-spacing:0 !important;
  font-family: var(--ov-sans) !important; color: var(--ov-steel) !important; font-size: 12.5px !important; }
.wrap-inner, .secondary-wrap { background: transparent !important; border: none !important; }
.tabitem .block { padding-left: 0 !important; padding-right: 0 !important; }
.tabitem .row, .ov-settings .row, .tabitem .form, .ov-settings .form { gap: 18px !important; }
.ov-settings .row { border: none !important; background: transparent !important; padding: 4px 0 !important; }
.ov-settings .block { padding: 8px 10px !important; border: none !important; background: transparent !important; }
.ov-settings .column { min-width: 0 !important; gap: 4px !important; }
.ov-settings .row > * { flex: 1 1 0 !important; min-width: 0 !important; }
.tabitem .block.ov-out, .tabitem .block:has(> .wrap > .upload-container), .tabitem .block:has(audio),
.tabitem .block:has([data-testid=waveform-controls]) { padding: 8px !important; border: 1px solid var(--ov-border) !important;
  background: var(--ov-well) !important; }
.ov-settings .ov-reset { align-self: end; }
textarea, input[type=text], input[type=number], select, .wrap:has(> .wrap-inner) {
  background: var(--ov-well) !important; color: var(--ov-text) !important;
  border: 1px solid var(--ov-border) !important; border-radius: 2px !important;
  font-family: var(--ov-sans) !important; }
textarea:focus, input:focus { border-color: #2f5a35 !important; outline: none !important; }
textarea::placeholder, input::placeholder { color: #6f7d72 !important; opacity: 1; }
input[type=number] { width: 100% !important; text-align: left; -moz-appearance: textfield; }
.head input[type=number] { min-width: 76px !important; width: auto !important; text-align: center !important; }
input[type=number]::-webkit-inner-spin-button, input[type=number]::-webkit-outer-spin-button {
  -webkit-appearance: none; margin: 0; }
input[type=range] { accent-color: var(--ov-green) !important; }
input[type=checkbox] { appearance: none !important; -webkit-appearance: none !important;
  width: 17px !important; height: 17px !important; border: 1px solid #4f7754 !important;
  background: var(--ov-well) !important; border-radius: 2px !important; position: relative; cursor: pointer; }
input[type=checkbox]:checked { background: var(--ov-green) !important; border-color: var(--ov-green) !important; }
input[type=checkbox]:checked::after { content:""; position:absolute; left:5px; top:1px; width:4px; height:9px;
  border: solid var(--ov-body); border-width: 0 2px 2px 0; transform: rotate(45deg); }
input[type=radio] { accent-color: var(--ov-green) !important; }
.wrap label, label.svelte-1bx8sav { background: transparent !important; }
ul.options, .options { background: var(--ov-raised) !important; border: 1px solid var(--ov-border) !important; }
ul.options li { color: var(--ov-text) !important; }
ul.options li.selected, ul.options li:hover, ul.options li.active { background: var(--ov-teal) !important; }

/* buttons */
button { font-family: var(--ov-mono) !important; letter-spacing: .08em; text-transform: uppercase;
  border-radius: 2px !important; font-size: 12.5px !important; font-weight: 400 !important; }
button.primary, .primary { background: var(--ov-green) !important; color: #041204 !important;
  border: 1px solid var(--ov-green) !important; font-weight: 500 !important; }
.ov-settings .ov-checks { gap: 14px !important; padding-top: 6px; }
button.primary:hover { background: var(--ov-green-hi) !important; }
button.secondary, button.sm { background: var(--ov-raised) !important; color: var(--ov-text) !important;
  border: 1px solid var(--ov-border) !important; }
button.secondary:hover { border-color: #2f5a35 !important; color: var(--ov-bone) !important; }
button.stop { background: transparent !important; color: var(--ov-fault) !important;
  border: 1px solid #5a2a33 !important; }
.tabitem button.lg, .runbar button { width: 100% !important; }
button:disabled { opacity: .55; animation: ovpulse 1.4s ease-in-out infinite; }
@keyframes ovpulse { 0%,100% { opacity: .45 } 50% { opacity: .8 } }

/* accordion */
.ov-settings { border: 1px solid var(--ov-border) !important; background: rgb(13 18 14 / 0.92) !important;
  margin-top: 14px !important; }
.ov-settings > button, .label-wrap { font-family: var(--ov-mono) !important; color: var(--ov-steel) !important;
  letter-spacing: .1em; text-transform: uppercase; font-size: 12px !important; }

/* audio */
.ov-out { border: 1px solid var(--ov-border) !important; background: var(--ov-well) !important; }
audio { width: 100%; }
.waveform-container, .component-wrapper { background: transparent !important; }

/* console */
#ov-console textarea { font-family: var(--ov-mono) !important; font-size: 12px !important;
  color: var(--ov-ash) !important; background: var(--ov-well) !important; line-height: 1.55; }
.ov-status textarea { font-family: var(--ov-mono) !important; font-size: 12.5px !important;
  color: var(--ov-text) !important; }

/* guide */
.ov-guide h3 { font-family: var(--ov-mono) !important; color: var(--ov-green) !important; font-weight: 400 !important;
  font-size: 13px !important; letter-spacing: .14em; text-transform: uppercase; margin: 22px 0 8px !important; }
.ov-guide p, .ov-guide li { color: var(--ov-text) !important; font-size: 14.5px; line-height: 1.6; }
.ov-guide code { background: var(--ov-well) !important; color: var(--ov-bone) !important;
  border: 1px solid var(--ov-border); padding: 1px 5px; border-radius: 2px; font-family: var(--ov-mono); }
.ov-tip { border: 1px solid var(--ov-border); border-left: 2px solid var(--ov-green); background: var(--ov-well);
  padding: 10px 14px; margin: 10px 0; color: var(--ov-ash); font-size: 14px; }
.prose, .md { color: var(--ov-text) !important; }
.ov-foot { font-family: var(--ov-mono); font-size: 11.5px; color: var(--ov-steel); letter-spacing: .06em;
  padding: 12px 16px; border: 1px solid var(--ov-border); margin: 14px 0 28px; display: flex;
  align-items: center; justify-content: space-between; background: rgb(6 9 6 / 0.92); }
/* one solid dock for run bar + console so no board strips show between them */
.ov-dock { background: rgb(13 18 14 / 0.92) !important; border: 1px solid var(--ov-border) !important;
  padding: 14px 16px !important; margin-top: 14px !important; gap: 12px !important; }
.ov-dock .row, .ov-dock .block, .ov-dock .form { background: transparent !important; border: none !important; }
.ov-dock .row { gap: 12px !important; }
/* header/footer HTML blocks flush with the panels; tab bar sits directly on its panel */
.html-container:has(> .prose > .ov-foot), .html-container:has(> .prose > .ov-head) {
  padding-left: 0 !important; padding-right: 0 !important; }
.ov-foot { margin-bottom: 0 !important; }
.tab-wrapper { padding-bottom: 0 !important; height: auto !important; margin-bottom: 0 !important; }
.tab-container { height: auto !important; }
"""


def _render_css():
    css = _CSS_TEMPLATE
    for k, v in C.items():
        css = css.replace(f"%({k})s", v)
    return font_faces() + css


CSS = _render_css()


def make_theme():
    t = gr.themes.Base(
        primary_hue=gr.themes.colors.green, neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.Font("Inter Tight"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.Font("JetBrains Mono"), "Consolas", "monospace"],
        radius_size=gr.themes.sizes.radius_none,
    )
    pairs = dict(
        body_background_fill=C["body"], body_text_color=C["text"], body_text_color_subdued=C["steel"],
        background_fill_primary=C["panel"], background_fill_secondary=C["well"],
        block_background_fill=C["panel"], block_border_color=C["border"], block_label_text_color=C["steel"],
        block_title_text_color=C["steel"], block_label_background_fill="transparent",
        border_color_primary=C["border"], border_color_accent=C["green"],
        input_background_fill=C["well"], input_border_color=C["border"], input_placeholder_color=C["fog"],
        button_primary_background_fill=C["green"], button_primary_text_color="#041204",
        button_primary_background_fill_hover=C["green_hi"], button_primary_border_color=C["green"],
        button_secondary_background_fill=C["raised"], button_secondary_text_color=C["text"],
        button_secondary_border_color=C["border"], button_secondary_background_fill_hover=C["teal"],
        slider_color=C["green"], checkbox_background_color_selected=C["green"],
        color_accent=C["green"], color_accent_soft=C["teal"], link_text_color=C["green"],
        panel_background_fill=C["panel"], panel_border_color=C["border"],
        table_even_background_fill=C["panel"], table_odd_background_fill=C["raised"],
        code_background_fill=C["well"], shadow_drop="none", shadow_drop_lg="none", block_shadow="none",
    )
    kw = {}
    for k, v in pairs.items():
        kw[k] = v
        kw[k + "_dark"] = v
    try:
        return t.set(**kw)
    except TypeError:
        # drop keys this Gradio version doesn't know
        import inspect

        ok = set(inspect.signature(t.set).parameters)
        return t.set(**{k: v for k, v in kw.items() if k in ok})


_HEAD_BASE = """
<script>
document.documentElement.lang = 'en';
(() => { const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'dark') { u.searchParams.set('__theme','dark'); window.location.replace(u); } })();
</script>
"""


def _board_head():
    """Board photo (inlined, ~290 KB) + trace polylines + the animation script."""
    try:
        js = open(os.path.join(ROOT, "assets", "board.js"), encoding="utf-8").read()
        traces = open(os.path.join(ROOT, "assets", "traces.json"), encoding="utf-8").read()
        img = base64.b64encode(open(os.path.join(ROOT, "assets", "board.jpg"), "rb").read()).decode()
    except OSError as e:
        log(f"Board background disabled ({e}).")
        return ""
    js = js.replace("__BOARD_SRC__", "data:image/jpeg;base64," + img).replace("__TRACES__", traces.strip())
    return "<script>" + js + "</script>"


FORCE_DARK = _HEAD_BASE + _board_head()

# ---------------------------------------------------------------------------
# Guide copy
# ---------------------------------------------------------------------------
GUIDE = """
<div class="ov-guide">

### Start here
The model loads the first time you generate, which takes a minute and downloads about 3.3 GB on the very first run. If you'd rather load it up front, go to the MODEL tab and press Load model. Everything you make is saved as a WAV file, with a small JSON file next to it that records the settings, so you can always find out how a clip was made.

### Cloning a voice
Give it a clean clip of 3 to 10 seconds with one person talking and no music behind them. Longer clips don't help: they slow generation down and can make the result worse. If you type the exact words spoken in the clip into the transcript box, cloning is more accurate. Leave it empty and Whisper will write it for you, or press Transcribe to fill it in and correct it yourself.

<div class="ov-tip">For the most natural result, make the reference clip in the same language as the text you want spoken. A Spanish clip reading English text will carry a Spanish accent, and sometimes that's exactly what you want.</div>

Type a name in "Save as voice" and the encoded voice is kept in the voices folder. Next time, switch the source to Saved voice and it skips the encoding step. The batch tab can use saved voices too.

### Designing a voice
Pick attributes from the dropdowns, or type them as a comma separated list such as `female, elderly, low pitch, british accent`. You only need one; the model fills in the rest. Accents only work on English text and dialects only on Chinese text. Voice design was trained on Chinese and English, so other languages can come out unstable. If an attribute seems to be ignored, use fewer of them.

### Expressive tags and pronunciation
You can drop sound tags into the text, such as `[laughter]` or `[sigh]`. The Insert tag menu adds one to the end of your text, and you can then move it where you need it. For English words that come out wrong, write the pronunciation in CMU phonemes inside brackets, for example `[B EY1 S]` for the bass in bass guitar. Chinese takes pinyin with tone numbers, like `ZHE2`.

### Numbers
Digits are read one by one unless you turn on Normalize numbers in the generation settings, or write them out as words. Normalizing covers whole numbers in many languages but not all of them. If yours isn't covered, write numbers out as words.

### Generation settings
These apply to every tab. The defaults are good, so change one thing at a time.

- Steps: 32 is the default quality. 16 is roughly twice as fast with a small loss.
- Guidance: how closely the output follows the voice and text. Around 2 is right; very high values sound strained.
- Speed and Duration: Speed stretches or squeezes the pace. Duration forces an exact length in seconds and overrides Speed. If the length must be exact, also turn off Trim silence.
- Seed: -1 picks a new random seed each time. The seed used is shown after every run, so you can type it back in to repeat a result.
- Position and class temperature: more randomness. Class temperature 0 is the safe default.
- Long text is split into chunks automatically, so you can paste a whole page. The chunk settings control where that happens.

### Batch
One line becomes one clip. All clips use the same voice and settings and are saved in a dated batch folder, with a zip ready to download. A larger batch size is faster but uses more VRAM.

### Model
You can point the app at a different checkpoint or a fine-tuned folder, or apply a LoRA adapter folder that holds `adapter_config.json`. The settings are remembered. Unload the model when you need the GPU for something else; anything else holding VRAM slows generation down or makes it fail.

### Use it responsibly
Only clone voices you have permission to use. Don't use this for impersonation, fraud or anything that deceives people.

</div>
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
SETTING_DEFAULTS = [32, 2.0, 0.1, 5.0, 0.0, 5.0, True, True, True, 15, 30, 0.1, 0.1, 1.0, None, False, -1]


def settings_panel():
    with gr.Accordion("Generation settings · shared by every tab", open=False, elem_classes="ov-settings", elem_id="ov-settings"):
        with gr.Row():
            steps = gr.Slider(4, 64, value=32, step=1, label="Steps", info="16 is about 2x faster, 32 is the default.")
            guidance = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance (CFG)", info="Default 2.0.")
            speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed", info="Above 1 is faster.")
            duration = gr.Number(value=None, label="Duration (s)", placeholder="empty = auto", info="Exact length. Overrides Speed.")
        with gr.Row():
            t_shift = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Time shift", info="Default 0.1.")
            pos_temp = gr.Slider(0.0, 10.0, value=5.0, step=0.1, label="Position temperature", info="0 is deterministic. Default 5.")
            cls_temp = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Class temperature", info="0 is greedy. Default 0.")
            layer_pen = gr.Slider(0.0, 10.0, value=5.0, step=0.1, label="Layer penalty", info="Default 5.")
        with gr.Row():
            chunk_dur = gr.Slider(5, 30, value=15, step=1, label="Chunk length (s)", info="Long text is cut into pieces this long.")
            chunk_thr = gr.Slider(10, 120, value=30, step=1, label="Chunk above (s)", info="Cutting starts past this length.")
            pad = gr.Slider(0.0, 0.5, value=0.1, step=0.01, label="Edge padding (s)", info="Silence added at each end.")
            fade = gr.Slider(0.0, 0.5, value=0.1, step=0.01, label="Fade (s)", info="Fade in and out.")
        with gr.Row():
            seed = gr.Number(value=-1, precision=0, label="Seed", info="-1 = random each run.")
            with gr.Column(elem_classes="ov-checks"):
                denoise = gr.Checkbox(value=True, label="Clean speech", info="Asks the model for denoised output.")
                pre = gr.Checkbox(value=True, label="Clean reference", info="Trims silence from the reference clip.")
            with gr.Column(elem_classes="ov-checks"):
                post = gr.Checkbox(value=True, label="Trim silence", info="Removes long pauses from the output.")
                normalize = gr.Checkbox(value=False, label="Normalize numbers", info="Reads 2345 as words.")
            with gr.Column(elem_classes="ov-reset"):
                reset = gr.Button("Reset to defaults", variant="secondary")
    comps = [steps, guidance, t_shift, pos_temp, cls_temp, layer_pen, denoise, pre, post,
             chunk_dur, chunk_thr, pad, fade, speed, duration, normalize, seed]
    reset.click(lambda: SETTING_DEFAULTS, outputs=comps)
    return comps



def text_box(placeholder):
    return gr.Textbox(label="Text to speak", lines=6, max_lines=20, placeholder=placeholder)


def lang_dd():
    return gr.Dropdown(LANGUAGES, value="Auto", label="Text language", info="Language of the text you type. Auto works; naming it helps a little.",
                       filterable=True)


TAG_PICK = "pick a tag"


def tag_dd():
    return gr.Dropdown([TAG_PICK] + NONVERBAL_TAGS, value=TAG_PICK, label="Insert tag",
                       info="Adds a sound tag at the end of the text.")


def output_col(prefix):
    audio = gr.Audio(label="Output", type="filepath", interactive=False, elem_classes="ov-out")
    status = gr.Textbox(label="Status", lines=2, interactive=False, elem_classes="ov-status",
                        placeholder="Results show up here.")
    return audio, status


def busy(label):
    return lambda: gr.update(value="Working…", interactive=False)


def free(label):
    return lambda: gr.update(value=label, interactive=True)


def build():
    voices = list_voices()
    s = load_settings()
    with gr.Blocks(title="OmniVoice · AIINFLUENCE") as demo:
        header = gr.HTML(header_html())

        with gr.Tabs():
            # 01 CLONE ---------------------------------------------------
            with gr.Tab("01 · Clone", elem_id="tab-clone"):
                gr.HTML(eyebrow("01", "Voice clone",
                                "Copy a voice from a short clip. Use a new reference clip or a voice you saved earlier."))
                with gr.Row():
                    with gr.Column():
                        c_text = text_box("What should the cloned voice say?")
                        with gr.Row():
                            c_lang = lang_dd()
                            c_tag = tag_dd()
                        c_src = gr.Radio(["New reference clip", "Saved voice"], value="New reference clip",
                                         label="Voice source")
                        with gr.Column(visible=True) as c_new:
                            c_ref = gr.Audio(label="Reference clip · 3 to 10 s", type="filepath",
                                             sources=["upload", "microphone"])
                            c_reftext = gr.Textbox(label="Reference transcript", lines=2,
                                                   placeholder="Exact words spoken in the clip. Leave empty to auto-transcribe.")
                            c_trans = gr.Button("Transcribe the clip", variant="secondary")
                            c_save = gr.Textbox(label="Save as voice (optional)",
                                                placeholder="name it to reuse later, e.g. narrator_calm")
                        with gr.Column(visible=False) as c_old:
                            with gr.Row():
                                c_voice = gr.Dropdown(voices, value=voices[0] if voices else None,
                                                      label="Saved voice")
                                c_vref = gr.Button("Refresh", variant="secondary", size="sm")
                            c_vprev = gr.Audio(label="Reference preview", type="filepath", interactive=False)
                            c_vtext = gr.Textbox(label="Stored transcript", interactive=False, lines=2)
                    with gr.Column():
                        c_instr = gr.Textbox(label="Extra style (optional)", lines=1,
                                             placeholder="e.g. whisper, low pitch")
                        c_out, c_status = output_col("c")
                        c_btn = gr.Button("Generate", variant="primary", size="lg")

            # 02 DESIGN --------------------------------------------------
            with gr.Tab("02 · Design", elem_id="tab-design"):
                gr.HTML(eyebrow("02", "Voice design",
                                "Describe the speaker with attributes. No reference audio needed."))
                with gr.Row():
                    with gr.Column():
                        d_text = text_box("What should the designed voice say?")
                        with gr.Row():
                            d_lang = lang_dd()
                            d_tag = tag_dd()
                        d_picks = []
                        cats = list(DESIGN_CATEGORIES.items())
                        for i in range(0, len(cats), 3):
                            with gr.Row():
                                for cat, opts in cats[i:i + 3]:
                                    d_picks.append(gr.Dropdown([ANY] + opts, value=ANY, label=cat,
                                                               info=DESIGN_INFO.get(cat)))
                        d_free = gr.Textbox(label="Or type attributes", lines=1,
                                            placeholder="female, young adult, high pitch, british accent")
                        d_preview = gr.Textbox(label="Instruction sent to the model", interactive=False, lines=1,
                                               placeholder="Builds itself as you pick attributes.")
                    with gr.Column():
                        d_out, d_status = output_col("d")
                        d_btn = gr.Button("Generate", variant="primary", size="lg")

            # 03 AUTO ----------------------------------------------------
            with gr.Tab("03 · Auto", elem_id="tab-auto"):
                gr.HTML(eyebrow("03", "Auto voice",
                                "The model picks a voice on its own. Fix the seed in settings to get the same voice back."))
                with gr.Row():
                    with gr.Column():
                        a_text = text_box("Anything you like. Long text is split automatically.")
                        with gr.Row():
                            a_lang = lang_dd()
                            a_tag = tag_dd()
                    with gr.Column():
                        a_out, a_status = output_col("a")
                        a_btn = gr.Button("Generate", variant="primary", size="lg")

            # 04 BATCH ---------------------------------------------------
            with gr.Tab("04 · Batch", elem_id="tab-batch"):
                gr.HTML(eyebrow("04", "Batch",
                                "One line of text, one clip. Every clip uses the same voice and settings."))
                with gr.Row():
                    with gr.Column():
                        b_lines = gr.Textbox(label="Lines", lines=10, max_lines=30,
                                             placeholder="First line becomes 001.wav\nSecond line becomes 002.wav")
                        with gr.Row():
                            b_lang = lang_dd()
                            b_bs = gr.Slider(1, 16, value=4, step=1, label="Batch size",
                                             info="More at once is faster, uses more VRAM.")
                        b_src = gr.Radio(["Saved voice", "Designed voice", "Auto voice"], value="Auto voice",
                                         label="Voice")
                        with gr.Column(visible=False) as b_saved_box:
                            b_voice = gr.Dropdown(voices, value=voices[0] if voices else None, label="Saved voice")
                            b_vref = gr.Button("Refresh voice list", variant="secondary")
                        b_picks = []
                        with gr.Column(visible=False) as b_design_box:
                            with gr.Row():
                                for cat in ("Gender", "Age", "Pitch"):
                                    b_picks.append(gr.Dropdown([ANY] + DESIGN_CATEGORIES[cat], value=ANY, label=cat))
                            with gr.Row():
                                for cat in ("Style", "English accent", "Chinese dialect"):
                                    b_picks.append(gr.Dropdown([ANY] + DESIGN_CATEGORIES[cat], value=ANY, label=cat))
                            b_free = gr.Textbox(label="Or type attributes", lines=1, placeholder="male, middle-aged")
                        b_auto_note = gr.HTML('<p class="stage-copy">The model picks a voice. Fix the seed in the settings below to keep the same voice across runs.</p>')
                    with gr.Column():
                        b_out = gr.Audio(label="First clip", type="filepath", interactive=False, elem_classes="ov-out")
                        b_zip = gr.File(label="All clips (zip)", interactive=False)
                        b_status = gr.Textbox(label="Status", lines=2, interactive=False, elem_classes="ov-status")
                        b_btn = gr.Button("Generate batch", variant="primary", size="lg")

            # 05 HISTORY -------------------------------------------------
            with gr.Tab("05 · History", elem_id="tab-history"):
                gr.HTML(eyebrow("05", "History",
                                "Every clip you generate, newest first, with the settings that made it."))
                with gr.Row():
                    with gr.Column():
                        hist = history_choices()
                        h_pick = gr.Dropdown(hist, value=hist[0][1] if hist else None, label="Clip", filterable=True)
                        with gr.Row():
                            h_ref = gr.Button("Refresh", variant="secondary")
                            h_open = gr.Button("Open outputs folder", variant="secondary")
                        h_audio = gr.Audio(label="Playback", type="filepath", interactive=False, elem_classes="ov-out")
                        h_msg = gr.Textbox(label="Status", lines=1, interactive=False, elem_classes="ov-status")
                    with gr.Column():
                        h_meta = gr.Code(label="Settings used", language="json", lines=22, interactive=False)

            # 06 MODEL ---------------------------------------------------
            with gr.Tab("06 · Model", elem_id="tab-model"):
                gr.HTML(eyebrow("06", "Model",
                                "Choose the checkpoint, precision and an optional LoRA adapter. Settings are remembered."))
                with gr.Row():
                    with gr.Column():
                        m_id = gr.Textbox(label="Checkpoint", value=s["model"],
                                          info="Hugging Face repo id or a local folder.")
                        with gr.Row():
                            m_dev = gr.Dropdown(["cuda", "cpu"], value=s["device"], label="Device")
                            m_dtype = gr.Dropdown(["float16", "bfloat16", "float32"], value=s["dtype"], label="Precision",
                                                  info="float16 is the tested default on NVIDIA.")
                        m_lora = gr.Textbox(label="LoRA adapter folder", value=s["lora_adapter"],
                                            placeholder="optional: folder with adapter_config.json")
                        m_asr = gr.Textbox(label="Whisper model for transcripts", value=s["asr_model"])
                        m_pre_asr = gr.Checkbox(value=False, label="Load Whisper now",
                                                info="Otherwise it loads the first time a transcript is needed.")
                        with gr.Row():
                            m_load = gr.Button("Load model", variant="primary")
                            m_unload = gr.Button("Unload", variant="stop")
                        m_status = gr.Textbox(label="Status", lines=2, interactive=False, elem_classes="ov-status")
                    with gr.Column():
                        gr.HTML(
                            '<div class="ov-guide">'
                            '<h3>Footprint</h3><p>The base checkpoint is about 2.5 GB plus an 0.8 GB audio tokenizer. '
                            'Whisper for auto-transcripts adds about 1.6 GB when it loads.</p>'
                            '<h3>LoRA</h3><p>An adapter is merged into the model when it loads. To go back to the base '
                            'model, clear the field and load again.</p>'
                            '<h3>Precision</h3><p>Keep float16 on NVIDIA cards. Use float32 only on CPU, where it is '
                            'chosen automatically.</p></div>'
                        )

            # GUIDE ------------------------------------------------------
            with gr.Tab("How to use", elem_id="tab-guide"):
                gr.HTML(eyebrow("··", "Guide", "How to get good results out of each tab."))
                gr.Markdown(GUIDE)

        settings = settings_panel()

        with gr.Column(elem_classes="ov-dock"):
            with gr.Row(elem_classes="runbar"):
                refresh_btn = gr.Button("Refresh console", variant="secondary")
                clear_btn = gr.Button("Clear console", variant="secondary")
            console = gr.Textbox(label="Console", lines=8, max_lines=8, interactive=False, elem_id="ov-console",
                                 value=console_text(), autoscroll=True)
        gr.HTML(
            '<div class="ov-foot"><span>OMNIVOICE · K2-FSA · APACHE-2.0 &nbsp;/&nbsp; OUTPUTS IN ./outputs '
            '&nbsp;/&nbsp; VOICES IN ./voices &nbsp;/&nbsp; AIINFLUENCE</span>'
            '<span>BOARD<button class="ov-fx is-on" title="Animated board background">FX ON</button></span></div>'
        )

        # ---- wiring -----------------------------------------------------
        def chain(btn, label, fn, inputs, outputs):
            return btn.click(busy(label), outputs=btn).then(fn, inputs=inputs, outputs=outputs).then(
                free(label), outputs=btn).then(lambda: (header_html(), console_text()), outputs=[header, console])

        c_src.change(lambda v: (gr.update(visible=v != "Saved voice"), gr.update(visible=v == "Saved voice")),
                     c_src, [c_new, c_old])
        c_voice.change(h_voice_info, c_voice, [c_vprev, c_vtext])
        c_vref.click(h_refresh_voices, c_voice, c_voice)
        c_trans.click(h_transcribe, c_ref, [c_reftext, c_status]).then(console_text, outputs=console)
        for txt, tg in ((c_text, c_tag), (d_text, d_tag), (a_text, a_tag)):
            tg.input(append_tag, [txt, tg], [txt, tg])
        chain(c_btn, "Generate", h_clone,
              [c_text, c_lang, c_src, c_voice, c_ref, c_reftext, c_save, c_instr] + settings,
              [c_out, c_status]).then(h_refresh_voices, c_voice, c_voice).then(
                  h_refresh_voices, b_voice, b_voice)

        for comp in d_picks + [d_free]:
            comp.change(design_instruct, [d_free] + d_picks, d_preview)
        chain(d_btn, "Generate", h_design, [d_text, d_lang, d_free] + d_picks + settings, [d_out, d_status])
        chain(a_btn, "Generate", h_auto, [a_text, a_lang] + settings, [a_out, a_status])

        b_vref.click(h_refresh_voices, b_voice, b_voice)
        b_src.change(lambda v: (gr.update(visible=v == "Saved voice"), gr.update(visible=v == "Designed voice"),
                                gr.update(visible=v == "Auto voice")), b_src, [b_saved_box, b_design_box, b_auto_note])
        chain(b_btn, "Generate batch", h_batch,
              [b_lines, b_lang, b_src, b_voice, b_free] + b_picks + [b_bs] + settings, [b_out, b_zip, b_status])

        h_ref.click(h_history_refresh, outputs=h_pick)
        h_pick.change(h_history_pick, h_pick, [h_audio, h_meta])
        h_open.click(h_open_outputs, outputs=h_msg)

        chain(m_load, "Load model", h_load_model, [m_id, m_dev, m_dtype, m_asr, m_lora, m_pre_asr],
              [m_status, header])
        m_unload.click(h_unload, outputs=[m_status, header]).then(console_text, outputs=console)

        refresh_btn.click(console_text, outputs=console)

        def _clear():
            LOG.clear()
            return console_text()

        clear_btn.click(_clear, outputs=console)

        timer = gr.Timer(3.0)
        timer.tick(lambda: (header_html(), console_text()), outputs=[header, console])
        demo.load(h_voice_info, c_voice, [c_vprev, c_vtext])
        demo.load(h_history_pick, h_pick, [h_audio, h_meta])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7870)
    ap.add_argument("--listen", action="store_true", help="Serve on 0.0.0.0 (local network).")
    ap.add_argument("--preload", action="store_true", help="Load the model at startup.")
    ap.add_argument("--open", action="store_true", help="Open the browser.")
    args = ap.parse_args()
    if args.preload:
        s = load_settings()
        try:
            ENGINE.load(s["model"], s["device"], s["dtype"], s["asr_model"], s["lora_adapter"])
        except Exception as e:
            log(f"Preload failed: {type(e).__name__}: {e}")
    demo = build()
    demo.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0" if args.listen else "127.0.0.1",
        server_port=args.port,
        inbrowser=args.open,
        theme=make_theme(),
        css=CSS,
        head=FORCE_DARK,
        allowed_paths=[OUTPUTS_DIR, VOICES_DIR],
    )


if __name__ == "__main__":
    main()
