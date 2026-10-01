"""Serve the app on :7872 with ONE neutral demo clip in a temp outputs folder (CPU, GPU untouched).

    .venv/Scripts/python.exe tools/demo_server.py      # then run readme_shots.py against :7872
"""
import os, sys, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import app

tmp = tempfile.mkdtemp(prefix="ov_demo_")
app.OUTPUTS_DIR = tmp
app.VOICES_DIR = tempfile.mkdtemp(prefix="ov_demo_voices_")
app.ENGINE.load("k2-fsa/OmniVoice", "cpu", "float32", None, "")
S = [32, 2.0, 0.1, 5.0, 0.0, 5.0, True, True, True, 15, 30, 0.1, 0.1, 1.0, None, False, 42]
picks = ["female", "young adult", "any", "any", "british accent", "any"]
print(app.h_design("Welcome to the voice lab. Everything you hear was made on this machine.", "English", "", *picks, *S), flush=True)
app.ENGINE.unload(quiet=True)
app.LOG.clear()
app.log("Ready.")
demo = app.build()
demo.queue().launch(server_name="127.0.0.1", server_port=7872, theme=app.make_theme(), css=app.CSS,
                    head=app.FORCE_DARK, allowed_paths=[tmp])
