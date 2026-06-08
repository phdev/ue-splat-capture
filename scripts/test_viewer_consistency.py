"""Viewer consistency test. Run before every Pages commit.

Catches the class of bugs where a new scene gets DEPLOYED (sog + settings.json
exist in out/site/) but never gets WIRED into the dropdown in index.html — so
the file is technically live but invisible to anyone using the dropdown UI.
That class of bug bit us with scene25: I shipped scene25.sog + scene25.json,
the default sogName pointed at scene25, but I never added scene25 to the
SCENES array — so it appeared "not in the dropdown" even though the file was
on the CDN.

Checks:
  1. Every scene*.sog in out/site/ has a SCENES[] entry in index.html
  2. Every SCENES[] entry has a matching .sog file on disk
  3. Every SCENES[] settings: path is a real .json file on disk
  4. The default sogName ('const sogName = ... || "sceneX.sog"') points at a
     file that exists AND is listed in SCENES[] (else dropdown can't select it)
  5. The "DEFAULT" suffix in SCENES[] labels matches the default sogName

Exit code: 0 = pass, 1 = any check failed (with a useful report).

  python3 scripts/test_viewer_consistency.py
"""
import os
import re
import sys

SITE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "site")
INDEX_HTML = os.path.join(SITE_DIR, "index.html")


def parse_index_html(html: str):
    """Pull (default_sog, [{file, label, settings}, ...]) from index.html."""
    # Default sog: const sogName = url.searchParams.get('content') || "sceneX.sog";
    m = re.search(r"""const\s+sogName\s*=\s*url\.searchParams\.get\(['"]content['"]\)\s*\|\|\s*["']([^"']+)["']""", html)
    default_sog = m.group(1) if m else None

    # SCENES entries: { file: 'scene...', label: '...', settings: 'scene...' }
    # Match each entry independently so a malformed one doesn't drop neighbors.
    entries = []
    for m in re.finditer(
        r"""\{\s*file:\s*['"]([^'"]+)['"]\s*,\s*label:\s*['"]([^'"]+)['"]\s*,\s*settings:\s*['"]([^'"]+)['"]\s*\}""",
        html,
    ):
        entries.append({"file": m.group(1), "label": m.group(2), "settings": m.group(3)})
    return default_sog, entries


def main() -> int:
    if not os.path.isfile(INDEX_HTML):
        print(f"FAIL: missing {INDEX_HTML}")
        return 1
    html = open(INDEX_HTML).read()
    default_sog, entries = parse_index_html(html)
    print(f"default sog: {default_sog}")
    print(f"dropdown entries: {len(entries)}")

    sogs_on_disk = sorted(f for f in os.listdir(SITE_DIR) if f.startswith("scene") and f.endswith(".sog"))
    listed_files = {e["file"] for e in entries}
    fails = []

    # 1. Every scene*.sog on disk is in the dropdown
    for f in sogs_on_disk:
        if f not in listed_files:
            fails.append(f"orphan: {f} exists on disk but is NOT in SCENES[] (won't appear in dropdown)")

    # 2 + 3. Every listed entry has its .sog and .json files
    for e in entries:
        sog_path = os.path.join(SITE_DIR, e["file"])
        json_path = os.path.join(SITE_DIR, e["settings"])
        if not os.path.isfile(sog_path):
            fails.append(f"dangling: SCENES entry references missing file {e['file']}")
        if not os.path.isfile(json_path):
            fails.append(f"dangling: SCENES entry references missing settings {e['settings']}")

    # 4. Default sog: file exists AND is in dropdown
    if default_sog is None:
        fails.append("could not parse default sogName from index.html")
    else:
        if not os.path.isfile(os.path.join(SITE_DIR, default_sog)):
            fails.append(f"default sogName='{default_sog}' but the file is missing")
        if default_sog not in listed_files:
            fails.append(f"default sogName='{default_sog}' is NOT in SCENES[] — dropdown can't reach it")

    # 5. DEFAULT label suffix matches the actual default
    default_labeled = [e for e in entries if "DEFAULT" in e["label"].upper()]
    if len(default_labeled) > 1:
        fails.append(f"multiple entries labeled DEFAULT: {[e['file'] for e in default_labeled]}")
    elif len(default_labeled) == 1 and default_labeled[0]["file"] != default_sog:
        fails.append(
            f"DEFAULT label is on {default_labeled[0]['file']} but default sogName is {default_sog}"
        )

    if fails:
        print(f"\nFAIL ({len(fails)} issue(s)):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nOK — viewer is internally consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
