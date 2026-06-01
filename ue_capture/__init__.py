"""ue_capture -- the Unreal-side package (runs inside UnrealEditor-Cmd's Python).

Only `rig.py` and `detect.py` are import-safe outside Unreal (pure Python). The
modules that touch the `unreal` API (`selftest_scene.py`, `render.py`,
`export.py`, `run_capture.py`) import `unreal` lazily so this package can still
be imported (e.g. for unit-testing the rig) in a normal venv.
"""
