"""Script to run ruff format on the git diff."""

import argparse
import sys

sys.path.append(__file__.split("scripts")[0])

from ruff_utils import (
    apply_patch,
    filter_patch_by_diff,
    get_diff_line_no,
    get_ruff_commands,
    print_files_to_reformat,
    print_patch,
    ruff_format_diff,
)


def main() -> None:  # noqa: C901
    """Main script to run ruff format on the git diff."""
    # Create parser.
    ruff_commands = get_ruff_commands("format")
    parser = argparse.ArgumentParser(description="Format using ruff.")
    parser.add_argument(
        "files",
        nargs=argparse.ZERO_OR_MORE,
        default=["."],
        help="List of files or directories to format",
    )
    parser.add_argument(
        "--git-diff",
        default="HEAD",
        type=str,
        help=(
            "Git diff to compare against, i.e. argument to pass to git diff"
            " (default: HEAD)"
        ),
    )
    ruff_commands["check"].add_as_argument(parser)
    ruff_commands["diff"].add_as_argument(parser)
    direct_ruff_flags: list[str] = []
    for cmd_name in (
        "target-version",
        "extension",
    ):
        ruff_commands[cmd_name].add_as_argument(parser)
        direct_ruff_flags.append(cmd_name)
    exclude = {
        "respect-gitignore",
        "force-exclude",
    }
    for group in (
        "Miscellaneous",
        "File selection",
        "Format configuration",
        "Editor options",
    ):
        argument_group = parser.add_argument_group(group)
        for cmd in ruff_commands.values():
            if cmd.name in exclude:
                continue
            if cmd.group == group:
                cmd.add_as_argument(argument_group)
                direct_ruff_flags.append(cmd.name)

    # Parse arguments.
    args = parser.parse_args()
    ruff_args = {
        arg: value
        for arg in direct_ruff_flags
        if (value := getattr(args, arg.replace("-", "_"))) is not None
    }
    ruff_flags: list[str] = []
    for arg, value in ruff_args.items():
        if isinstance(value, str):
            ruff_flags.append(f"--{arg} {value}")
        elif value:
            ruff_flags.append(f"--{arg}")

    # Obtain diff ranges and format patch.
    diff_no = get_diff_line_no(args.git_diff)
    patch = ruff_format_diff(args.files, *ruff_flags)
    filtered_patch = filter_patch_by_diff(patch, diff_no)

    # Print info and optionally apply patch.
    n_reformatted_files = len(filtered_patch)
    if n_reformatted_files == 0:
        s = "s" if len(patch) > 1 else ""
        print(f"{len(patch)} file{s} left unchanged")
        raise SystemExit(0)
    s = "s" if n_reformatted_files > 1 else ""
    match args.check, args.diff:
        case True, False:
            print_files_to_reformat(filtered_patch)
            print(f"{n_reformatted_files} file{s} would be reformatted")
        case False, True:
            print_patch(filtered_patch)
            print(f"{n_reformatted_files} file{s} would be reformatted")
        case False, False:
            apply_patch(filtered_patch)
            print(f"{n_reformatted_files} file{s} reformatted")
    raise SystemExit(int(args.check and n_reformatted_files > 0))


if __name__ == "__main__":
    main()
