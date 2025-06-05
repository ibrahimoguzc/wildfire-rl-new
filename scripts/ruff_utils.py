"""Utilies for running ruff check and format on the git diff."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from git import Repo
from ruff.__main__ import find_ruff_bin
from unidiff import LINE_TYPE_ADDED, Hunk, PatchedFile, PatchSet

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

__all__ = [
    "Command",
    "Violation",
    "ViolationLocation",
    "filter_violations_by_diff",
    "get_diff_line_no",
    "get_ruff_commands",
    "print_violations",
    "ruff_check_violations",
]

RUFF_BIN = find_ruff_bin()
REPO = Repo(search_parent_directories=True)
REPO_PATH = Path(REPO.working_dir)
RE_REFORMAT = re.compile(r"[0-9]+ files? would be reformatted")


def posix_filename(filename: str | Path) -> str:
    """Get the filename as a POSIX path relative to the repository."""
    return Path(filename).resolve().relative_to(REPO_PATH).as_posix()


@dataclass(slots=True)
class ViolationLocation:
    """A Ruff violation location."""

    column: int
    row: int


@dataclass(slots=True, kw_only=True)
class Violation:
    """A Ruff violation."""

    cell: str | None
    code: str
    end_location: ViolationLocation
    filename: str
    fix: dict | None
    location: ViolationLocation
    message: str
    noqa_row: int
    url: str

    def __post_init__(self) -> None:
        """Post init method."""
        self.filename = posix_filename(self.filename)
        if isinstance(self.end_location, dict):
            self.end_location = ViolationLocation(**self.end_location)
        if isinstance(self.location, dict):
            self.location = ViolationLocation(**self.location)


def bold(s: str) -> str:
    """Make string bold."""
    return f"\033[1m{s}\033[0m"


def red(s: str) -> str:
    """Make string red."""
    return f"\033[91m{s}\033[0m"


def green(s: str) -> str:
    """Make string green."""
    return f"\033[92m{s}\033[0m"


def dark_blue(s: str) -> str:
    """Make string dark blue."""
    return f"\033[34m{s}\033[0m"


def light_blue(s: str) -> str:
    """Make string light blue."""
    return f"\033[96m{s}\033[0m"


def violation_to_msg_summary(violation: Violation) -> str:
    """Get a summary of the violation."""
    v = violation
    colon = light_blue(":")
    return (
        f"{bold(v.filename)}{colon}{v.location.row}{colon}"
        f"{v.location.column}{colon} {red(v.code)} {v.message}"
    )


def violation_to_msg_full(violation: Violation) -> str:
    """Convert violation to message."""
    v = violation
    message_lines = [violation_to_msg_summary(v)]
    with Path(v.filename).open() as f:
        source_lines = f.readlines()
    start_idx = max(0, v.location.row - 3)
    end_idx = min(len(source_lines), v.end_location.row + 2)
    if v.location.row == v.end_location.row:  # Single line.
        message_lines.append(dark_blue(f"{'':4} |"))
        for i, line in enumerate(
            source_lines[start_idx : v.location.row], start=start_idx + 1
        ):
            message_lines.append(dark_blue(f"{i:4} | ") + line.rstrip())
        message_lines.append(
            dark_blue(f"{'':4} | ")
            + red(
                " " * (v.location.column - 1)
                + "^" * (v.end_location.column - v.location.column)
                + f" {v.code}"
            )
        )
        for i, line in enumerate(
            source_lines[v.end_location.row : end_idx],
            start=v.end_location.row + 1,
        ):
            message_lines.append(dark_blue(f"{i:4} | ") + line.rstrip())
        message_lines.append(dark_blue(f"{'':4} |"))
    else:  # Multi-line.
        message_lines.append(dark_blue(f"{'':4} |"))
        for i, line in enumerate(
            source_lines[start_idx : v.location.row - 1], start=start_idx + 1
        ):
            message_lines.append(dark_blue(f"{i:4} |   ") + line.rstrip())
        message_lines.append(
            dark_blue(f"{v.location.row:4} | ")
            + red("/ ")
            + source_lines[v.location.row - 1].rstrip()
        )
        for i, line in enumerate(
            source_lines[v.location.row : v.end_location.row],
            start=v.location.row + 2,
        ):
            message_lines.append(
                dark_blue(f"{i:4} |") + red(" | ") + line.rstrip()
            )
        message_lines.append(dark_blue(f"{'':4} | ") + red(f"|_^ {v.code}"))
        for i, line in enumerate(
            source_lines[v.end_location.row : end_idx],
            start=v.end_location.row + 2,
        ):
            message_lines.append(dark_blue(f"{i:4} |   ") + line.rstrip())
        message_lines.append(dark_blue(f"{'':4} |"))
    return "\n".join(message_lines)


@dataclass(slots=True, kw_only=True)
class Command:
    """A Ruff command."""

    name: str
    group: str
    value_format: str | None = field(default=None)
    description: str

    def add_as_argument(self, parser: argparse.ArgumentParser) -> None:
        """Add command as an argument to the parser."""
        if self.value_format is None:
            parser.add_argument(
                f"--{self.name}",
                action="store_true",
                help=self.description,
            )
        else:
            parser.add_argument(
                f"--{self.name}",
                nargs="?",
                type=str,
                help=self.description,
            )


def get_diff_line_no(reference: str) -> dict[str, list[int]]:
    """Get diff line numbers.

    Args:
        reference: Reference to compare against. Example: "HEAD".

    Returns:
        Dictionary of diff numbers ({filename: [line_no]}).
    """
    return get_patch_line_no(PatchSet(REPO.git.diff(reference)))


def get_hunk_line_no(hunk: Hunk) -> list[int]:
    """Get the diff line numbers of a hunk."""
    return [
        line.target_line_no
        for line in hunk
        if line.line_type == LINE_TYPE_ADDED
    ]


def get_patch_line_no(patch_set: PatchSet) -> dict[str, list[int]]:
    """Get diff range of a PatchSet."""
    return {
        posix_filename(patched_file.path): [
            line for hunk in patched_file for line in get_hunk_line_no(hunk)
        ]
        for patched_file in patch_set
    }


def _call_ruff(
    command: Literal["check", "format"], files: Sequence[str], *args: str
) -> str:
    """Run ruff check or format."""
    command = [RUFF_BIN, command, *(" ".join((*files, *args)).split())]
    out = subprocess.run(
        command, capture_output=True, text=True, check=False, shell=False
    )
    return out.stdout


def get_ruff_commands(command: str) -> dict[str, Command]:
    """Get ruff commands."""
    group_re = re.compile(r"\n([\w ]+):\n")
    cmd_re = re.compile(r"--([\w-]+)(?: <([\w_]+)>)?[\t\s]+(.*?)\n")
    help_text = _call_ruff(command, (), "--help") + "\nEmpty end group:\n"
    commands: dict[str, Command] = {}
    for g1, g2 in pairwise(group_re.finditer(help_text)):
        group_name = g1.group(1)
        group_help = help_text[g1.end() : g2.start()].strip()
        # Strip intermediate "\n    " parts that don't end with "-""
        group_help = re.sub(r"\n\s+(?=[^\s-])", " ", group_help)
        for m in cmd_re.finditer(group_help):
            name, value_format, description = m.groups()
            commands[name] = Command(
                name=name,
                group=group_name,
                value_format=value_format,
                description=description,
            )
    return commands


def ruff_check_violations(files: Sequence[str], *args: str) -> list[Violation]:
    """Run ruff check on files.

    Args:
        files: Files to check.
        args: Additional arguments to pass to ruff check.

    Returns:
        List of all violations.
    """
    result = _call_ruff("check", files, *args, "--output-format=json")
    if not result:
        return []
    return [Violation(**violation) for violation in json.loads(result)]


def ruff_check_fix(files: Sequence[str], *args: Sequence[str]) -> PatchSet:
    """Run ruff check obtaining fixes."""
    result = _call_ruff("check", files, *args, "--diff")
    if not result:
        return PatchSet([])
    # Remove last line which is the summary.
    return PatchSet("\n".join(result.splitlines()[:-1]))


def ruff_format_diff(files: Sequence[str], *args: Sequence[str]) -> PatchSet:
    result = _call_ruff("format", files, *args, "--diff")
    if not result:
        return PatchSet([])
    # Remove last line which is the summary.
    return PatchSet("\n".join(result.splitlines()[:-1]))


def _is_within_range(rng: tuple[int, int], nos: list[int]) -> bool:
    """Check if a number is in a range."""
    return any(rng[0] <= x <= rng[1] for x in nos)


def filter_violations_by_diff(
    violations: list[Violation], diff_no: dict[str, list[int]]
) -> list[Violation]:
    """Filter violations by diff ranges.

    Args:
        violations: List of violations.
        diff_no: Dictionary of diff ranges ({filename: [line_no]}).

    Returns:
        List of violations that are in the diff ranges.
    """
    return [
        v
        for v in violations
        if _is_within_range(
            (v.location.row, v.end_location.row), diff_no.get(v.filename, [])
        )
    ]


def filter_patch_by_diff(
    patch_set: PatchSet,
    diff_ranges: dict[str, list[int]],
) -> PatchSet:
    """Filter patch by diff ranges.

    Args:
        patch_set: PatchSet to filter.
        diff_ranges: Dictionary of diff ranges ({filename: [line_no]}).

    Returns:
        Filtered PatchSet.
    """
    patched_files = []
    for patched_file in patch_set:
        diff = set(diff_ranges.get(posix_filename(patched_file.path), []))
        if not diff:
            continue
        filtered_hunks = [
            hnk
            for hnk in patched_file
            if diff.intersection(get_hunk_line_no(hnk))
        ]
        if filtered_hunks:
            new_patched_file = deepcopy(patched_file)
            new_patched_file.clear()
            new_patched_file.extend(filtered_hunks)
            patched_files.append(new_patched_file)
    return PatchSet.from_string(
        "\n".join(str(patched_file) for patched_file in patched_files)
    )


def add_ab_prefixes(patch: PatchSet | PatchedFile) -> None:
    """Add 'a/' and 'b/' prefixes to source and target files.

    Adds 'a/' and 'b/' prefixes to source and target files if they are
    missing. These prefixes are required by git apply.
    """
    if isinstance(patch, PatchSet):
        for patched_file in patch:
            add_ab_prefixes(patched_file)
        return
    if not (
        patch.source_file.startswith("a/")
        and patch.target_file.startswith("b/")
    ):
        patch.source_file = (Path("a") / patch.source_file).as_posix()
        patch.target_file = (Path("b") / patch.target_file).as_posix()


def apply_patch(patch: PatchSet | PatchedFile) -> None:
    """Apply patch."""
    if len(patch) == 0:
        return
    patch = deepcopy(patch)
    add_ab_prefixes(patch)
    # Create temporary patch file.
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(str(patch) + "\n")
        patch_file = f.name
    out = subprocess.run(
        [REPO.git.GIT_PYTHON_GIT_EXECUTABLE, "apply", patch_file],
        check=False,
        capture_output=True,
    )
    Path(patch_file).unlink()
    if out.stderr:
        raise ValueError(out.stderr)


def print_violations(
    violations: list[Violation],
    *,
    output_format: Literal["full", "consice"] = "full",
) -> None:
    """Print violations."""
    if output_format == "full":
        for v in violations:
            print(violation_to_msg_full(v), end="\n\n")
    elif output_format == "consice":
        for v in violations:
            print(violation_to_msg_summary(v))
    if violations:
        s = "" if len(violations) == 1 else "s"
        print(f"Found {len(violations)} error{s}.")
    else:
        print("All checks passed!")


def print_patch(patch: PatchSet) -> None:
    """Print patch."""
    lines = str(patch).splitlines()
    for i, line in enumerate(lines):
        if line.startswith("---"):
            lines[i] = "---" + red(line[3:])
        elif line.startswith("+++"):
            lines[i] = "+++" + green(line[3:])
        elif line.startswith("-"):
            lines[i] = red(line)
        elif line.startswith("+"):
            lines[i] = green(line)
    print("\n".join(lines))


def print_files_to_reformat(patch: PatchSet) -> None:
    """Print files to reformat."""
    for patched_file in patch:
        print(f"Would reformat: {bold(patched_file.path)}")
