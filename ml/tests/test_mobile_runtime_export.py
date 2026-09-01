import subprocess
from pathlib import Path

import pytest

from crrc_vision.mobile_runtime_export import (
    build_cmake_build_command,
    build_cmake_configure_command,
    build_mnn_command,
    build_pnnx_command,
    normalize_captured_output,
    validate_checkout,
)


def test_mnn_command_is_path_explicit(tmp_path: Path) -> None:
    converter = tmp_path / "MNN Convert.exe"
    model = tmp_path / "model input.onnx"
    output = tmp_path / "model output.mnn"
    converter.write_bytes(b"exe")
    model.write_bytes(b"onnx")

    assert build_mnn_command(converter, model, output) == [
        str(converter.resolve()),
        "-f",
        "ONNX",
        "--modelFile",
        str(model.resolve()),
        "--MNNModel",
        str(output.resolve()),
        "--bizCode",
        "crrc-fastener",
    ]


def test_mnn_command_can_disable_graph_optimizations(tmp_path: Path) -> None:
    converter = tmp_path / "MNNConvert.exe"
    model = tmp_path / "model.onnx"
    output = tmp_path / "model.mnn"
    converter.write_bytes(b"exe")
    model.write_bytes(b"onnx")

    command = build_mnn_command(
        converter,
        model,
        output,
        optimize_level=0,
    )

    assert command[-2:] == ["--optimizeLevel", "0"]


def test_pnnx_command_names_all_outputs_below_run(tmp_path: Path) -> None:
    converter = tmp_path / "pnnx.exe"
    model = tmp_path / "model.onnx"
    output = tmp_path / "ncnn"
    converter.write_bytes(b"exe")
    model.write_bytes(b"onnx")

    command = build_pnnx_command(converter, model, output, fp16=True)

    assert command[:4] == [
        str(converter.resolve()),
        str(model.resolve()),
        "inputshape=[1,3,640,640]",
        "fp16=1",
    ]
    assert f"ncnnparam={(output / 'model.ncnn.param').resolve()}" in command
    assert f"ncnnbin={(output / 'model.ncnn.bin').resolve()}" in command


def test_pnnx_command_accepts_a_512_square_input(tmp_path: Path) -> None:
    converter = tmp_path / "pnnx.exe"
    model = tmp_path / "model.onnx"
    output = tmp_path / "ncnn"
    converter.write_bytes(b"exe")
    model.write_bytes(b"onnx")

    command = build_pnnx_command(
        converter,
        model,
        output,
        fp16=False,
        input_size=512,
    )

    assert command[2] == "inputshape=[1,3,512,512]"
    assert command[3] == "fp16=0"


def test_cmake_commands_use_explicit_source_build_and_target(tmp_path: Path) -> None:
    cmake = tmp_path / "cmake.exe"
    source = tmp_path / "source"
    build = tmp_path / "build"
    cmake.write_bytes(b"exe")
    source.mkdir()

    configure = build_cmake_configure_command(
        cmake=cmake,
        source=source,
        build=build,
        definitions={"MNN_BUILD_CONVERTER": "ON", "MNN_BUILD_SHARED_LIBS": "OFF"},
    )
    compile_command = build_cmake_build_command(cmake, build, "MNNConvert")

    assert configure[:6] == [
        str(cmake.resolve()),
        "-S",
        str(source.resolve()),
        "-B",
        str(build.resolve()),
        "-G",
    ]
    assert configure[-2:] == [
        "-DMNN_BUILD_CONVERTER=ON",
        "-DMNN_BUILD_SHARED_LIBS=OFF",
    ]
    assert compile_command == [
        str(cmake.resolve()),
        "--build",
        str(build.resolve()),
        "--target",
        "MNNConvert",
        "--config",
        "Release",
    ]


def test_checkout_must_match_pinned_revision(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Test"],
        check=True,
    )
    (checkout / "file.txt").write_text("content", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "file.txt"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-q", "-m", "test"], check=True)

    with pytest.raises(RuntimeError, match="RUNTIME_REVISION_MISMATCH"):
        validate_checkout(checkout, "deadbeef")


def test_captured_process_output_stays_binary_safe() -> None:
    assert normalize_captured_output(None) == b""
    assert normalize_captured_output(b"valid\xffbinary") == b"valid\xffbinary"
