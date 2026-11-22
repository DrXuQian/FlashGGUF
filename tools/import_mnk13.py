"""Import the verified offline kit into an EMPTY destination, preserving every source byte."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

KIT_SHA = "0271b71d894144d27e7445ed4d4848c79337faf290bbb6f5363c460eb93fc1de"
PREFIX = "flashgguf-mnk13/"
ENTRIES = {"README.md", "run.py", "plan.json", "identity.json"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kit", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    if hashlib.sha256(a.kit.read_bytes()).hexdigest() != KIT_SHA:
        p.error("not the verified MNK13 box kit")
    a.dest.mkdir(exist_ok=False)
    archive = a.dest / "source.tar.gz"
    seen = set()
    with tarfile.open(a.kit, "r:gz") as src, archive.open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as gz, tarfile.open(fileobj=gz, mode="w|") as out:
            for m in src:
                if not m.isfile() or not m.name.startswith(PREFIX):
                    raise ValueError(m.name)
                key = m.name[len(PREFIX):]
                name = PurePosixPath(key)
                if name.is_absolute() or ".." in name.parts or key in seen:
                    raise ValueError(key)
                seen.add(key)
                with src.extractfile(m) as data:
                    if key in ENTRIES:
                        (a.dest/key).write_bytes(data.read())
                    elif name.parts[0] == "source" and len(name.parts) >= 2:
                        entry = tarfile.TarInfo(name.relative_to("source").as_posix())
                        entry.size = m.size
                        entry.mode = 0o644
                        out.addfile(entry, data)
                    else:
                        raise ValueError(key)
    inventory = json.loads((a.dest/"identity.json").read_bytes())["files"]
    if seen != set(inventory) | {"identity.json"}:
        raise ValueError("frozen kit inventory differs")
    result = hashlib.sha256(archive.read_bytes()).hexdigest()
    (a.dest/"source.sha256").write_text(result + "  source.tar.gz\n")


if __name__ == "__main__":
    main()
