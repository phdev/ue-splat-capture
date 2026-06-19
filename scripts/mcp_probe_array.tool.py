import json

IM = ("/Game/Levels/PCG/ElectricDreams_PCGCloseRange."
      "ElectricDreams_PCGCloseRange:PersistentLevel."
      "BP_PathFly_C_UAID_F02F4B078FB99FE502_1282371579.InterpMove")


def cp(n, base=0):
    return [{"positionControlPoint": {"x": (base + i) * 100.0, "y": 0.0, "z": 0.0},
             "bPositionIsRelative": True} for i in range(n)]


def setcp(arr):
    try:
        execute_tool("editor_toolset.toolsets.object.ObjectTools.set_properties",
                     json.dumps({"instance": {"refPath": IM},
                                 "values": json.dumps({"controlPoints": arr})}))
        return "ok"
    except Exception as e:
        return "ERR:" + str(e)[:60]


def count():
    r = execute_tool("editor_toolset.toolsets.object.ObjectTools.get_properties",
                     json.dumps({"instance": {"refPath": IM},
                                 "properties": ["controlPoints"]}))["returnValue"]
    return len(json.loads(r)["controlPoints"])


def run():
    res = {}
    res["start"] = count()
    res["set0_r"] = setcp([]); res["set0_n"] = count()
    res["set1_r"] = setcp(cp(1)); res["set1_n"] = count()
    res["set3_r"] = setcp(cp(3)); res["set3_n"] = count()
    res["set5_r"] = setcp(cp(5)); res["set5_n"] = count()
    res["set2_r"] = setcp(cp(2)); res["set2_n"] = count()
    return res
