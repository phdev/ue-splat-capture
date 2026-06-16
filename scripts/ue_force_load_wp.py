"""Force-load + PIN every World-Partition actor in the OPEN warm editor, then verify
instance counts. Run via scripts/ue_exec.py. Fixes "objects not enabled / missing" when
the cause is WP PARTIAL STREAMING (actors unloaded) -- distinct from hidden components
(set_visibility(False)); this script also reports hidden comps so you can tell which.

Close-range (ElectricDreams_PCGCloseRange) FULLY loaded = 738 ISMs / ~49,560 instances
(112 actors). A partial load (~17 actors / ~26K inst) means WP hasn't streamed -> this.
"""
import unreal
ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
world = ues.get_editor_world()

def counts():
    ismc = inst = hidden = 0
    for a in eas.get_all_level_actors():
        for c in a.get_components_by_class(unreal.InstancedStaticMeshComponent):
            cnt = c.get_instance_count(); ismc += 1; inst += cnt
            try:
                if not c.is_visible(): hidden += cnt
            except Exception:
                pass
    return len(eas.get_all_level_actors()), ismc, inst, hidden

a0, i0, n0, h0 = counts()
print(f"[wp] BEFORE level={world.get_name()} actors={a0} ISMC={i0} inst={n0} hidden_inst={h0}")
# viewport = streaming source near the island
try:
    unreal.EditorLevelLibrary.set_level_viewport_camera_info(unreal.Vector(90250, -4360, 9000), unreal.Rotator(-35, 0, 0))
except Exception as e:
    print("[wp] viewport:", e)
res = unreal.WorldPartitionBlueprintLibrary.get_actor_descs()
descs = res[1] if isinstance(res, (tuple, list)) else res
guids = []
for d in (descs or []):
    try: guids.append(d.get_editor_property("guid"))
    except Exception: pass
print(f"[wp] descriptors={len(guids)}")
if guids:
    unreal.WorldPartitionBlueprintLibrary.load_actors(guids)
    try: unreal.WorldPartitionBlueprintLibrary.pin_actors(guids); print("[wp] pinned")
    except Exception as e: print("[wp] pin:", e)
a1, i1, n1, h1 = counts()
print(f"[wp] AFTER  actors={a1} ISMC={i1} inst={n1} hidden_inst={h1}")
print("WP_FORCE_LOAD_DONE")
