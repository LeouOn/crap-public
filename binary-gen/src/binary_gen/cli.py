"""
Command-line interface for Binary Gen.

Generate, sign, convert, and execute raw binaries.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from binary_gen.cli_types import (
    DUMP_HEX_OPT,
    HEX_OPT,
    OUTPUT_FILE,
    PLATFORM_OPT,
    VERBOSE_OPT,
)
from binary_gen.cli_utils import configure_logging
from binary_gen.constants import DEFAULT_BIN_PATH, DEFAULT_MESSAGE
from binary_gen.core import (
    Platform,
    PlatformBuilder,
    get_builder,
    make_binary_executable,
    parse_hex,
    sign_binary,
)
from binary_gen.core.linux_oracle import METHODS, linux_oracle
from binary_gen.core.runner import Exited, Signaled, run_binary

MAX_MESSAGE_LENGTH = 64

_RUN_DESCRIPTION = (
    "Linux binaries run natively on Linux x86_64 hosts, otherwise in Docker (linux/amd64). "
    "macOS binaries execute natively and require a macOS ARM host."
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Binary Gen: Generate executable binaries from text messages.",
    add_completion=False,
)


@app.callback()
def _configure(
    verbose: VERBOSE_OPT = 0,
) -> None:
    """Configure logging based on the ``-v``/``-vv``/``-vvv`` flag count."""
    configure_logging(verbose)


def _validate_message(value: str) -> str:
    """Typer callback enforcing ASCII-only payloads within the byte-slot limit."""
    if not value.isascii():
        raise typer.BadParameter("Message must contain only ASCII characters.")

    if len(value) > MAX_MESSAGE_LENGTH:
        raise typer.BadParameter(f"Message must be {MAX_MESSAGE_LENGTH} characters or fewer.")

    return value


@app.command(name="gen")
def gen_cmd(
    message: Annotated[
        str,
        typer.Argument(
            help=f"The message to embed in the binary (max {MAX_MESSAGE_LENGTH} ASCII characters).",
            callback=_validate_message,
        ),
    ] = DEFAULT_MESSAGE,
    platform: PLATFORM_OPT = Platform.LINUX_X86,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output binary path.", **OUTPUT_FILE),
    ] = DEFAULT_BIN_PATH,
    run: Annotated[
        bool,
        typer.Option(
            help=f"Run the binary after building (similar to the 'run' command). {_RUN_DESCRIPTION}"
        ),
    ] = False,
    dump_hex: DUMP_HEX_OPT = True,
) -> None:
    """Generate an executable binary that prints `message` when run.

    Builds the binary for the target platform, writes it to `output`, applies
    the appropriate executable permissions (and code signature on macOS), and
    optionally executes it.
    """
    typer.secho("Starting build...", fg="cyan", bold=True)
    typer.echo(f"Payload: {message}")

    builder = get_builder(platform)

    binary_data = builder.build_print(message)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(binary_data)

    _finalize_binary(output, builder)
    _dump_file_info(output, dump_hex=dump_hex)

    if run:
        _run_binary(binary_data, platform)

    typer.secho(f"\nSuccess! Wrote executable binary to {output}", fg="green", bold=True)


@app.command(name="oracle")
def oracle_cmd(
    lines: Annotated[
        int, typer.Option(help="3 casts a trigram (exits 1-8), 6 a hexagram (exits 1-64).")
    ] = 6,
    method: Annotated[
        str, typer.Option(help=f"Line probabilities: {', '.join(METHODS)}.")
    ] = "coins",
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output binary path.", **OUTPUT_FILE),
    ] = DEFAULT_BIN_PATH,
    run: Annotated[
        bool,
        typer.Option(help=f"Cast once after building. {_RUN_DESCRIPTION}"),
    ] = False,
    dump_hex: DUMP_HEX_OPT = True,
) -> None:
    """Generate a Linux x86_64 I Ching oracle that casts a random figure.

    It prints the figure top line first (`x` and `o` mark changing lines)
    and exits with its King Wen (hexagram) or Earlier Heaven (trigram) number.
    """
    if method not in METHODS:
        raise typer.BadParameter(f"Method must be one of: {', '.join(METHODS)}.")
    try:
        binary_data = linux_oracle(lines, METHODS[method])
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(binary_data)

    _finalize_binary(output, get_builder(Platform.LINUX_X86))
    _dump_file_info(output, dump_hex=dump_hex)

    if run:
        _run_binary(binary_data, Platform.LINUX_X86)

    typer.secho(f"\nSuccess! Wrote oracle binary to {output}", fg="green", bold=True)


@app.command(name="run")
def run_cmd(
    source: Annotated[
        str,
        typer.Argument(
            help="Path to a binary or hex file, raw hex string, or '-' for stdin.",
        ),
    ] = str(DEFAULT_BIN_PATH),
    hex: HEX_OPT = False,
    platform: PLATFORM_OPT = Platform.LINUX_X86,
) -> None:
    """Execute a binary for the target platform.

    Linux binaries run natively on Linux x86_64 hosts, otherwise in Docker
    (linux/amd64). macOS binaries execute natively and require a macOS ARM host.

    Existing files are read as raw bytes; anything else is parsed as hex. Use
    ``--hex`` to force hex, e.g. for a file of hex text or stdin.
    """
    binary_data = _parse_input_as_hex_or_exit(source) if hex else _read_binary_input(source)
    _run_binary(binary_data, platform)


@app.command(name="from-hex")
def from_hex_cmd(
    source: Annotated[
        str,
        typer.Argument(help="Raw hex string, path to a hex file, or '-' for stdin."),
    ],
    platform: PLATFORM_OPT = Platform.LINUX_X86,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output binary path.", **OUTPUT_FILE),
    ] = DEFAULT_BIN_PATH,
    dump_hex: DUMP_HEX_OPT = True,
    run: Annotated[
        bool,
        typer.Option(
            help=f"Run the binary after converting. {_RUN_DESCRIPTION}",
        ),
    ] = False,
) -> None:
    """Convert a raw hex string back into a runnable binary."""
    binary_data = _parse_input_as_hex_or_exit(source)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(binary_data)

    builder = get_builder(platform)
    _finalize_binary(output, builder)
    _dump_file_info(output, dump_hex=dump_hex)

    if run:
        _run_binary(binary_data, platform)

    typer.secho(f"\nSuccess! Wrote executable binary to {output}", fg="green", bold=True)


def _parse_input_as_hex_or_exit(raw_input: str) -> bytes:
    """Read hex from a file, stdin or the literal argument, then parse it."""
    return _parse_hex_or_exit(_parse_text_input(raw_input))


def _parse_text_input(raw_input: str) -> str:
    """Resolve a CLI argument that may be a literal string, a file path, or stdin ('-').

    Long hex payloads passed directly on the command line can exceed the
    OS's maximum path length; we treat any `OSError` from the `is_file`
    probe as "not a path" and fall through to returning the raw string.
    """
    if raw_input == "-":
        return sys.stdin.read()

    try:
        candidate = Path(raw_input)

        if candidate.is_file():
            return candidate.read_text()

    except OSError:
        pass

    return raw_input


def _parse_hex_or_exit(hex_data: str) -> bytes:
    """Parse a hex string into bytes, or exit with a user-friendly error."""
    try:
        return parse_hex(hex_data)
    except ValueError:
        typer.secho("Error: Invalid hex string.", fg="red", bold=True)
        raise typer.Exit(code=1) from None


def _read_binary_input(source: str) -> bytes:
    """Read source as raw bytes if it's a file, otherwise parse as hex."""
    if source == "-":
        return sys.stdin.buffer.read()

    try:
        path = Path(source)

        if path.is_file():
            return path.read_bytes()

    except OSError:
        pass

    return _parse_hex_or_exit(source)


def _finalize_binary(output_path: Path, builder: PlatformBuilder) -> None:
    """Make the binary executable and, on macOS, sign it.

    The builder's placeholder signature fixes the LC_CODE_SIGNATURE layout,
    but the kernel won't run it until `codesign` replaces it with a valid
    ad-hoc signature. Linux needs only the chmod.
    """
    try:
        make_binary_executable(output_path)

        if builder.platform is Platform.MAC_ARM:
            sign_binary(output_path, builder.platform)

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        typer.secho(f"\nFailed to finalize binary: {e}", fg="red", bold=True)
        raise typer.Exit(code=1) from None


def _dump_file_info(output_path: Path, dump_hex: bool = True) -> None:
    """Print file type and size, and optionally dump a hex sidecar `.txt` file.

    Best-effort: missing system tools (`file`, `xxd`) emit a warning rather
    than aborting the command.
    """
    try:
        subprocess.run(["file", str(output_path)], check=True)
        typer.echo(f"Size: {output_path.stat().st_size} bytes")

        if dump_hex:
            txt_path = output_path.with_suffix(output_path.suffix + ".txt")

            with txt_path.open("w") as f:
                subprocess.run(["xxd", str(output_path)], stdout=f, check=True)

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("Inspection tools unavailable: %s", e)


def _run_binary(binary_data: bytes, platform: Platform) -> None:
    """Print stdout and distinguish nonzero exits from signal deaths."""
    result = run_binary(binary_data, platform)
    typer.echo(result.stdout.decode(errors="replace"), nl=False)

    match result.termination:
        case Exited(code=code) if code != 0:
            typer.secho(f"Exit code: {code}", fg="yellow")
        case Signaled(signal=number):
            typer.secho(f"Killed by signal: {number}", fg="red")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    """Console script entry point. Wired up in pyproject.toml."""
    app()


if __name__ == "__main__":
    main()
