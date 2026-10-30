"""Remove PNG ICC metadata without decoding/re-encoding any image pixels."""

from __future__ import annotations

import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

from tqdm import tqdm

from .common import discover_images

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _profile_ranges(path: Path) -> tuple[int, list[tuple[int, int]]]:
    """Locate whole iCCP chunks by seeking over payloads, using bounded memory."""
    size = path.stat().st_size
    ranges = []
    with path.open("rb") as stream:
        if stream.read(8) != PNG_SIGNATURE:
            raise ValueError(f"Invalid PNG signature: {path}")
        first, has_pixels = True, False
        while True:
            start = stream.tell()
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError(f"Truncated PNG or missing IEND: {path}")
            length, kind = struct.unpack(">I4s", header)
            end = start + 12 + length  # length + type + data + CRC
            if length > 0x7FFFFFFF or end > size:
                raise ValueError(f"Invalid/truncated PNG chunk {kind!r}: {path}")
            if first and (kind != b"IHDR" or length != 13):
                raise ValueError(f"Missing PNG IHDR: {path}")
            first = False
            if kind == b"iCCP":
                ranges.append((start, end))
            has_pixels |= kind == b"IDAT"
            if kind == b"IEND":
                if length != 0 or not has_pixels:
                    raise ValueError(f"Invalid PNG image structure: {path}")
                return size, ranges
            stream.seek(end)


def _copy_bytes(source, destination, count: int) -> None:
    while count:
        block = source.read(min(count, 1024 * 1024))
        if not block:
            raise ValueError("PNG changed or became truncated while copying")
        destination.write(block)
        count -= len(block)


def strip_png_icc(path: Path, *, in_place: bool = False) -> bool:
    """Return whether iCCP exists; optionally remove it via atomic replacement.

    All other bytes, including compressed pixels, CRCs, transparency, EXIF and
    any trailing data, are copied unchanged. No full image is held in memory.
    """
    path = Path(path)
    if in_place and path.is_symlink():
        raise ValueError(f"Refusing to replace a symbolic link: {path}")
    before = path.stat()
    size, ranges = _profile_ranges(path)
    if not ranges or not in_place:
        return bool(ranges)
    temporary = None
    try:
        with path.open("rb") as source, tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as destination:
            temporary = Path(destination.name)
            cursor = 0
            for start, end in ranges:
                _copy_bytes(source, destination, start - cursor)
                source.seek(end)
                cursor = end
            _copy_bytes(source, destination, size - cursor)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"PNG changed during cleaning; original left in place: {path}")
        shutil.copystat(path, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def run(args) -> int:
    images = [path for path in discover_images(args.images) if path.suffix.lower() == ".png"]
    if not images:
        raise ValueError(f"No PNG files found: {args.images}")
    summary = {"png_files": len(images), "with_icc": 0, "cleaned": 0, "unchanged": 0, "errors": 0}
    with tqdm(images, desc="Clean PNG ICC" if args.in_place else "Inspect PNG ICC",
              unit="image", dynamic_ncols=True) as progress:
        for path in progress:
            try:
                has_icc = strip_png_icc(path, in_place=args.in_place)
                summary["with_icc" if has_icc else "unchanged"] += 1
                if has_icc and args.in_place:
                    summary["cleaned"] += 1
            except (OSError, ValueError) as exc:
                summary["errors"] += 1
                tqdm.write(f"ERROR {path}: {exc}")
            progress.set_postfix(cleaned=summary["cleaned"], errors=summary["errors"], refresh=False)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not args.in_place:
        print("Inspection only. Add --in-place to remove iCCP metadata while preserving compressed pixel data.", flush=True)
    return 1 if summary["errors"] else 0
