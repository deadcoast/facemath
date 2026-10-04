"""Destination-side staging for outputs that will be handed to the caller.

Private tempfile directories retain restrictive Windows ACLs after rename.
Published results instead inherit the chosen parent directory's ordinary ACL
(and the process umask on POSIX). The caller owns cleanup and final rename.
"""
from pathlib import Path
from secrets import token_hex


def publication_stage(parent: Path, prefix: str) -> Path:
    """Reserve a random sibling directory without opening an existing path."""
    parent = Path(parent)
    if not prefix or Path(prefix).name != prefix or "/" in prefix or "\\" in prefix:
        raise ValueError("publication prefix must be a single path component")
    for _ in range(100):
        candidate = parent / (prefix + token_hex(16))
        try:
            candidate.mkdir(mode=0o777)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError("cannot reserve a publication staging directory")
