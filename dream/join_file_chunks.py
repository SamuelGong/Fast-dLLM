#!/usr/bin/env python3
"""
join_file_chunks.py

Join text file chunks (e.g. prefix_part1.txt, prefix_part2.txt, ...) into a single file.
"""

import os
import argparse
import re


def join_chunks(input_prefix, output_file):
    """
    Look for files named <input_prefix>_partN.txt in the current directory,
    ordered by N, and concatenate them into output_file.
    """
    pattern = re.compile(rf"^{re.escape(input_prefix)}_part(\d+)\.txt$")
    # Find matching files
    parts = []
    for fname in os.listdir('.'):
        m = pattern.match(fname)
        if m:
            parts.append((int(m.group(1)), fname))
    if not parts:
        print(f"No files found matching prefix '{input_prefix}_partN.txt'.")
        return
    # Sort by part index
    parts.sort(key=lambda x: x[0])

    with open(output_file, 'w') as outfile:
        for index, fname in parts:
            print(f"Appending {fname}...")
            with open(fname, 'r') as infile:
                for line in infile:
                    outfile.write(line)

    print(f"Joined {len(parts)} parts into '{output_file}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Join split text chunks into one file."
    )
    parser.add_argument(
        "input_prefix",
        help="Prefix used when splitting (e.g. 'myfile' for myfile_part1.txt, ...)."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Name of the joined output file (default: <prefix>_joined.txt)."
    )
    args = parser.parse_args()

    out_name = args.output if args.output else f"{args.input_prefix}_joined.txt"
    join_chunks(args.input_prefix, out_name)


if __name__ == '__main__':
    main()
