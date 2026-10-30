Built-in demo voice slot.

`app.py` copies `Aria.pt`, `Aria.json` and `Aria.wav` from this folder into
`voices/` on first launch, but only when those three files are present here.
Generate them once with:

    python tools/make_default_voice.py

(it loads the model, speaks a sample line, encodes the voice and writes all
three files here). Keep them in git so a fresh clone ships with a voice that
works out of the box.
