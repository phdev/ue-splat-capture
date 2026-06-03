#!/bin/zsh
# Headless Electric Dreams capture launcher. Requires the GUI editor CLOSED.
#   UE_PROBE=1 scripts/capture_headless_run.sh   # quick probe (load + overview)
#   UE_PROBE=0 scripts/capture_headless_run.sh   # full orbit capture
UE_CMD="/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor-Cmd"
PROJ="/Users/peterhowell/Documents/Unreal Projects/ElectricDreamsEnv/ElectricDreamsEnv.uproject"
SCRIPT="/Users/peterhowell/ue-splat-capture/ue_capture/capture_headless.py"
LEVEL="/Game/Levels/PCG/ElectricDreams_PCGCloseRange"
LOG="${UE_HL_LOG:-/tmp/ed_headless.log}"

# refuse to launch if the GUI editor still holds the project lock
if ps aux | grep -i "UnrealEditor.app/Contents/MacOS/UnrealEditor " | grep -v grep | grep -vq "Cmd"; then
  echo "ABORT: GUI UnrealEditor still running — quit it first (releases the project lock)." ; exit 3
fi

echo "[launch] PROBE=${UE_PROBE:-0} EV=${UE_CAPTURE_EV:-1.0} N_AZ=${UE_N_AZ:-28} -> log $LOG"
"$UE_CMD" "$PROJ" "$LEVEL" \
  -ExecutePythonScript="$SCRIPT" \
  -unattended -nosplash -nop4 -RenderOffScreen -NoSound -stdout -FullStdOutLogOutput \
  > "$LOG" 2>&1
echo "[launch] exit=$? (see $LOG)"
grep -E "PROBE_DONE|WROTE |\[hl\]" "$LOG" | tail -20
