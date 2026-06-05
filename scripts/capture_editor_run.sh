#!/bin/zsh
# Launch the GUI UnrealEditor (NOT -Cmd) and run the tick-driven capture
# (ue_capture/capture_editor.py). The GUI editor actually renders + ticks, so PCG
# foliage generates and TSR converges -- the things headless can't do. The python
# script registers a post-tick callback and returns; the capture runs as the editor
# ticks, then quit_editor() closes it. A window WILL open (that's required for real
# rendering). Pass UE_CAPTURE_OUT as an ABSOLUTE path.
# Launch the REAL .app binary directly. The bare Binaries/Mac/UnrealEditor is a stub
# that re-execs the .app copy and exits (exit=1 in ~4s), detaching the real editor so
# we lose its process + stdout. Targeting the .app binary keeps it in the foreground.
UE_CMD="/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
PROJ="/Users/peterhowell/Documents/Unreal Projects/ElectricDreamsEnv/ElectricDreamsEnv.uproject"
SCRIPT="/Users/peterhowell/ue-splat-capture/ue_capture/capture_editor.py"
LEVEL="${UE_LEVEL:-/Game/Levels/PCG/ElectricDreams_PCGCloseRange}"
LOG="${UE_HL_LOG:-/tmp/ed_editor.log}"

# refuse if a GUI editor already holds the project lock. Use pgrep -x on the exact
# executable name (the GUI binary is "UnrealEditor"; headless is "UnrealEditor-Cmd",
# not matched) so the check can't self-match other processes whose COMMAND LINE merely
# contains the editor path (the old `ps|grep` form false-positived and aborted).
if pgrep -x UnrealEditor >/dev/null 2>&1; then
  echo "ABORT: a GUI UnrealEditor is already running — quit it first." ; exit 3
fi

echo "[launch-gui] PROBE=${UE_PROBE:-0} EV=${UE_CAPTURE_EV:-10} -> log $LOG (a window will open)"
# NO -RenderOffScreen (we want real GPU rendering so PCG/TSR work). -stdout to capture log.
# Use -ExecCmds="py <script>" (console-command context) NOT -ExecutePythonScript:
# the latter is run-then-QUIT automation, so the editor exits the instant the script
# returns (before our tick callback ever fires). -ExecCmds runs it but leaves the
# editor open + ticking, so PCG generates and the tick-driven capture can proceed.
"$UE_CMD" "$PROJ" "$LEVEL" \
  -ExecCmds="py $SCRIPT" \
  -nosplash -nop4 -NoSound -stdout -FullStdOutLogOutput \
  > "$LOG" 2>&1
echo "[launch-gui] exit=$? (see $LOG)"
grep -E "EDITOR_CAPTURE_ARMED|EDITOR_CAPTURE_DONE|\[ed\]" "$LOG" | tail -20
