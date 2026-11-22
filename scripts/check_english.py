"""Language check: every tracked file must be English.

For each file of ``git ls-files`` the script counts the lines that contain an umlaut or sharp s, and
the lines that contain a word of a fixed German word list (on word boundaries, all lower case or with
a capital first letter; an all-caps token such as 'MIT' is no hit). It
prints each hit and exits with 1 when there is at least one hit, else with 0.

The script must not find its own word list. Thus each word is stored reversed and the non-ASCII
letters are written as escapes; the script reverses the words again when it starts.

Usage: python scripts/check_english.py [repo root]
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

_REVERSED_WORDS = (
    "dnu thcin redo driw nedrew tsi red eid sad hci ud dnis fua tim r\xfcf ettib gitref leiZ eleiZ "
    "efielhcS tratsueN gnudiehcstnE gnulhefpmE etsiL retiew bierhcs ereisuap ppots nehcerbba aj "
    "tssap tug sol tfu\xe4l"
)
WORDS = tuple(word[::-1] for word in _REVERSED_WORDS.split())
# Each word matches in 2 forms only: all lower case or with a capital first letter.
# An all-caps token (for example a licence name) is no hit.
FORMS = sorted({form for w in WORDS for form in (w.lower(), w[:1].upper() + w[1:].lower())})
UMLAUT = re.compile("[\xe4\xf6\xfc\xdf\xc4\xd6\xdc]")
WORD = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(w) for w in FORMS))


def tracked_files(root: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True)
    return [name for name in out.stdout.decode("utf-8").split("\0") if name]


def check(root: str) -> int:
    hits = 0
    for name in tracked_files(root):
        try:
            with open(os.path.join(root, name), encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(lines, start=1):
            umlaut = UMLAUT.search(line)
            word = WORD.search(line)
            if umlaut or word:
                hits += 1
                why = "umlaut" if umlaut else f"word {word.group(0)!r}"
                print(f"{name}:{number}: {why}: {line.strip()[:120]}")
    print(f"hits: {hits}")
    return 1 if hits else 0


if __name__ == "__main__":
    default_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(check(sys.argv[1] if len(sys.argv) > 1 else default_root))
