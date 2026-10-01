"""Backend smoke test: runs every generation handler in-process against the real model."""
import os, sys, time, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import app
import soundfile as sf

S = [16, 2.0, 0.1, 5.0, 0.0, 5.0, True, True, True, 15, 30, 0.1, 0.1, 1.0, None, False, 1234]
res = {}
t = time.time()
p, m = app.h_auto("Hello there. This is a quick test of the OmniVoice control panel.", "English", *S)
res["auto"] = (p, m); print("AUTO", m, flush=True)

picks = ["female", "young adult", "any", "any", "british accent", "any"]
p2, m2 = app.h_design("Good evening, and welcome to the voice lab.", "English", "", *picks, *S)
res["design"] = (p2, m2); print("DESIGN", m2, flush=True)

# clone from the design output, with known transcript, and save as voice
name = "smoke_test_voice"
for ext in (".pt", ".json", ".wav"):
    f = os.path.join(app.VOICES_DIR, name + ext)
    if os.path.exists(f): os.remove(f)
p3, m3 = app.h_clone("Now I am speaking with a cloned voice. It has 3 apples.", "English", "New reference clip", None,
                     p2, "Good evening, and welcome to the voice lab.", name, "", *S[:-2], True, S[-1])
res["clone"] = (p3, m3); print("CLONE", m3, flush=True)
print("VOICES", app.list_voices())

p4, m4 = app.h_clone("Reusing the saved voice from disk.", "English", "Saved voice", name, None, "", "", "", *S)
res["clone_saved"] = (p4, m4); print("CLONE_SAVED", m4, flush=True)

o1, z, m5 = app.h_batch("Line one of the batch.\nLine two, a little longer than the first.\nThird and last line.",
                        "English", "Saved voice", name, "", *(["any"] * 6), 2, *S)
res["batch"] = (o1, z, m5); print("BATCH", m5, flush=True)

print("NORM", app.normalize_numbers("I have 2345 apples [laughter].", "English"))
print("TOTAL", round(time.time() - t, 1), "s")
for k, v in res.items():
    path = v[0]
    if path:
        a, sr = sf.read(path)
        print(k, sr, round(len(a) / sr, 2), "s", "peak", round(float(abs(a).max()), 3))
    else:
        print(k, "NO OUTPUT")
