"""Script to run ruff check on the git diff."""

import argparse
import sys

sys.path.append(__file__.split("scripts")[0])

from ruff_utils import (
    apply_patch,
    filter_patch_by_diff,
    filter_violations_by_diff,
    get_diff_line_no,
    get_ruff_commands,
    print_patch,
    print_violations,
    ruff_check_fix,
    ruff_check_violations,
)


def main() -> None:  # noqa: C901, PLR0912
    """Main script to run ruff check on the git diff."""
    # Create parser.
    ruff_commands = get_ruff_commands("check")
    parser = argparse.ArgumentParser(description="Check format using ruff.")
    parser.add_argument(
        "files",
        nargs=argparse.ZERO_OR_MORE,
        default=["."],
        help="List of files or directories to check",
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
    ruff_commands["fix"].add_as_argument(parser)
    ruff_commands["diff"].add_as_argument(parser)
    supported_ruff_flags: list[str] = []
    for cmd_name in (
        "unsafe-fixes",
        "ignore-noqa",
        "target-version",
        "extension",
    ):
        ruff_commands[cmd_name].add_as_argument(parser)
        supported_ruff_flags.append(cmd_name)
    exclude = {
        "respect-gitignore",
        "force-exclude",
        "exit-zero",
        "exit-non-zero-on-fix",
    }
    for group in ("Rule selection", "File selection", "Miscellaneous"):
        argument_group = parser.add_argument_group(group)
        for cmd in ruff_commands.values():
            if cmd.name in exclude:
                continue
            if cmd.group == group:
                cmd.add_as_argument(argument_group)
                supported_ruff_flags.append(cmd.name)

    # Parse arguments.
    args = parser.parse_args()
    ruff_args = {
        arg: value
        for arg in supported_ruff_flags
        if (value := getattr(args, arg.replace("-", "_"))) is not None
    }
    ruff_flags: list[str] = []
    for arg, value in ruff_args.items():
        if isinstance(value, str):
            ruff_flags.append(f"--{arg} {value}")
        elif value:
            ruff_flags.append(f"--{arg}")

    # Check for ruff violations.
    diff_ranges = get_diff_line_no(args.git_diff)
    if args.fix or args.diff:
        patch = ruff_check_fix(args.files, *ruff_flags)
        filtered_patch = filter_patch_by_diff(patch, diff_ranges)
        if args.diff:
            print_patch(filtered_patch)
        else:
            apply_patch(filtered_patch)
    if not args.diff:
        violations = ruff_check_violations(args.files, *ruff_flags)
        filtered_violations = filter_violations_by_diff(
            violations, diff_ranges
        )
        print_violations(filtered_violations)
        if filtered_violations:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
