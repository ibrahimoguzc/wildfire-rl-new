# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

import sys

HEADER_LIMIT = 50  # Max characters in header
BODY_LINE_LIMIT = 72  # Max characters per line in body


def check_commit_msg(msg):
    lines = msg.split("\n")

    # Enforce header limit
    if len(lines[0]) > HEADER_LIMIT:
        return (
            f"Header is too long. It should be less than {HEADER_LIMIT} "
            "characters."
        )

    summary_started = False
    minimum_summary = False
    first_line_reached = False

    for line in lines[1:]:
        # Skip comments
        if line.startswith("#"):
            continue

        if not first_line_reached:
            if len(line) > 0:
                first_line_reached = True
                # Enforce indentation of body lines through length check of
                # first line.
                if len(line) <= HEADER_LIMIT:
                    first_line_reached = True
                    return (
                        "Body of commit message must be indented at 72 characters"
                        " and not 50 characters. ALT+Q can be used to shift"
                        " between the two indentation levels. "
                    )

        # Enforce Summary of Changes section
        if "Summary of Changes" in line:
            summary_started = True
            continue

        # Enforce minimum of one bullet point in Summary of Changes
        if summary_started and not minimum_summary:
            if len(line) > 0 and not line.startswith("- "):
                return (
                    'Lines after "Summary of Changes" should be bullet points '
                    'start with "- ").'
                )
            if len(line) > 0 and line.startswith("- "):
                minimum_summary = True

        # Enforce length of each line in the body
        if not len(line) <= BODY_LINE_LIMIT:
            return (
                f"Body lines should be less than {BODY_LINE_LIMIT} characters."
            )

    # Enforce Summary of Changes section
    if not summary_started:
        return '"Summary of Changes" section not found.'
    return None


def main():
    commit_msg_file = sys.argv[1]
    with open(commit_msg_file) as file:
        msg = file.read()

    error = check_commit_msg(msg)
    if error is not None:
        sys.stderr.write("Commit message error: " + error + "\n\n")
        sys.stderr.write("Modify your commit message: " + msg + "\n\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
