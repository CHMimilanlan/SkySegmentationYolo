"""PNG metadata cleanup preserves compressed chunks and decoded pixel values."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from skyseg import png_profiles
from skyseg.cli import parser
from skyseg.common import pixel_hash, read_rgb


def png_chunks(data):
    offset, chunks = 8, []
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        raw = data[offset:offset + 12 + length]
        chunks.append((raw[4:8], raw))
        offset += len(raw)
    return chunks


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "I;16"])
def test_strip_preserves_pixels_and_every_non_icc_chunk(tmp_path, mode):
    path = tmp_path / "image.png"
    if mode == "I;16":
        pixels = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32) * 67
    else:
        channels = 3 if mode == "RGB" else 4
        pixels = np.arange(24 * 32 * channels, dtype=np.uint8).reshape(24, 32, channels)
    image = Image.fromarray(pixels)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("note", "metadata that must survive")
    image.save(path, icc_profile=b"fixture ICC profile", pnginfo=metadata)
    before = path.read_bytes()
    before_chunks = png_chunks(before)
    assert any(kind == b"iCCP" for kind, _ in before_chunks)
    assert png_profiles.strip_png_icc(path, in_place=True)
    after = path.read_bytes()
    assert png_chunks(after) == [(kind, raw) for kind, raw in before_chunks if kind != b"iCCP"]
    with Image.open(path) as result:
        assert "icc_profile" not in result.info
        assert result.info["note"] == "metadata that must survive"
        np.testing.assert_array_equal(np.asarray(result), pixels)
    assert not png_profiles.strip_png_icc(path, in_place=True)
    assert path.read_bytes() == after  # second run leaves the file untouched


def test_read_rgb_export_does_not_reembed_inherited_profile(tmp_path):
    source, output = tmp_path / "source.png", tmp_path / "output.png"
    image = Image.new("RGB", (32, 24), (30, 50, 70))
    expected_hash = pixel_hash(image)
    image.save(source, icc_profile=b"fixture ICC profile")
    normalized = read_rgb(source)
    assert "icc_profile" not in normalized.info
    normalized.save(output)
    with Image.open(output) as result:
        assert "icc_profile" not in result.info
    assert pixel_hash(normalized) == expected_hash
    png_profiles.strip_png_icc(source, in_place=True)
    assert pixel_hash(read_rgb(source)) == expected_hash


def test_cli_inspects_by_default_and_only_modifies_with_flag(tmp_path, capsys):
    profiled, plain = tmp_path / "profiled.PNG", tmp_path / "plain.png"
    Image.new("RGB", (20, 16)).save(profiled, icc_profile=b"fixture ICC profile")
    Image.new("RGB", (20, 16)).save(plain)
    before = {path: path.read_bytes() for path in (profiled, plain)}
    args = parser().parse_args(["clean-png", "--images", str(tmp_path)])
    assert png_profiles.run(args) == 0
    assert all(path.read_bytes() == content for path, content in before.items())
    summary = json.loads(capsys.readouterr().out.splitlines()[0])
    assert summary["with_icc"] == 1 and summary["cleaned"] == 0
    args.in_place = True
    assert png_profiles.run(args) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[0])
    assert summary["cleaned"] == 1 and summary["unchanged"] == 1 and summary["errors"] == 0
    assert plain.read_bytes() == before[plain]


@pytest.mark.parametrize("kind", ["signature", "truncated"])
def test_malformed_png_is_left_unchanged(tmp_path, kind):
    path = tmp_path / "bad.png"
    Image.new("RGB", (20, 16)).save(path, icc_profile=b"fixture ICC profile")
    content = b"not a PNG" if kind == "signature" else path.read_bytes()[:-5]
    path.write_bytes(content)
    with pytest.raises(ValueError):
        png_profiles.strip_png_icc(path, in_place=True)
    assert path.read_bytes() == content
    assert not list(tmp_path.glob("*.tmp"))


def test_copy_failure_keeps_original_and_removes_temporary(tmp_path, monkeypatch):
    path = tmp_path / "image.png"
    Image.new("RGB", (20, 16)).save(path, icc_profile=b"fixture ICC profile")
    before = path.read_bytes()

    def fail_copy(*args):
        raise OSError("simulated storage error")

    monkeypatch.setattr(png_profiles, "_copy_bytes", fail_copy)
    with pytest.raises(OSError, match="simulated storage error"):
        png_profiles.strip_png_icc(path, in_place=True)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
