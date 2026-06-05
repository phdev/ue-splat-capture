#!/bin/zsh
# Launch the GUI UnrealEditor (NOT -Cmd) and run the tick-driven capture
# (ue_capture/capture_editor.py). The GUI editor actually renders + ticks, so PCG
# foliage generates and TSR converges -- the things headless can't do. The python
# script registers a post-tick callback and returns; the capture runs as the editor
# ticks, then quit_editor() closes it. A window WILL open (that's required for real
# rendering). Pass UE_CAPTURE_OUT as an ABSOLUTE path.
UE_CMD="/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor"   # GUI, not -Cmd
PROJ="/Users/peterhowell/Documents/Unreal Projects/ElectricDreamsEnv/ElectricDreamsEnv.uproject"
SCRIPT="/Users/peterhowell/ue-splat-capture/ue_capture/capture_editor.py"
LEVEL="${UE_LEVEL:-/Game/Levels/PCG/ElectricDreams_PCGCloseRange}"
LOG="${UE_HL_LOG:-/tmp/ed_editor.log}"

# refuse if a GUI editor already holds the project lock
if ps aux | grep -i "UnrealEditor.app/Contents/MacOS/UnrealEditor " | grep -v grep | grep -vq "Cmd"; then
  echo "ABORT: a GUI UnrealEditor is already running — quit it first." ; exit 3
fi

echo "[launch-gui] PROBE=${UE_PROBE:-0} EV=${UE_CAPTURE_EV:-10} -> log $LOG (a window will open)"
# NO -RenderOffScreen (we want real GPU rendering so PCG/TSR work). -stdout to capture log.
"$UE_CMD" "$PROJ" "$LEVEL" \
  -ExecutePythonScript="$SCRIPT" \
  -nosplash -nop4 -NoSound -stdout -FullStdOutLogOutput \
  > "$LOG" 2>&1
echo "[launch-gui] exit=$? (see $LOG)"
grep -E "EDITOR_CAPTURE_ARMED|EDITOR_CAPTURE_DONE|\[ed\]" "$LOG" | tail -20
