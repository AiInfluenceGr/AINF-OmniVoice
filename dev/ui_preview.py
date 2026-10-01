"""UI-only check: build the Blocks and serve on a side port without loading the model."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app

port = int(sys.argv[1]) if len(sys.argv) > 1 else 7871
demo = app.build()
demo.queue().launch(server_name="127.0.0.1", server_port=port, theme=app.make_theme(), css=app.CSS,
                    head=app.FORCE_DARK, allowed_paths=[app.OUTPUTS_DIR, app.VOICES_DIR], prevent_thread_lock=False)
