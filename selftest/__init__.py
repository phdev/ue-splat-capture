"""selftest -- a tiny, dependency-light stand-in for Unreal.

It defines a canonical scene (fiducials at KNOWN coordinates, simple diffuse
geometry, an orbit-hemisphere camera rig) in UE convention, and a numpy
raytracer that renders it. This is what produces the COMMITTED fixtures so the
whole verification suite (T1/T2/T3) reproduces without an Unreal install.

The forward projection here (`project_ue_native`) is written independently of
splatkit.convert, so the reprojection gate (T1) genuinely cross-checks the
coordinate conversion rather than comparing a function to itself.
"""
