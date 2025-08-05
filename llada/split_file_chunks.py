#!/usr/bin/env python3
"""
split_file_chunks.py

Split a text file into smaller files with a fixed number of lines per chunk.
"""

import os
import argparse


def split_file(input_file, lines_per_chunk=600, output_prefix=None):
    # Determine base name for output files
    if output_prefix:
        base_name = output_prefix
    else:
        base_name = os.path.splitext(os.path.basename(input_file))[0]

    chunk_index = 1
    line_count = 0
    output_file = None

    with open(input_file, 'r') as infile:
        for line in infile:
            # Start a new chunk when needed
            if line_count % lines_per_chunk == 0:
                if output_file:
                    output_file.close()
                chunk_filename = f"{base_name}_part{chunk_index}.txt"
                output_file = open(chunk_filename, 'w')
                print(f"Creating chunk file: {chunk_filename}")
                chunk_index += 1
            output_file.write(line)
            line_count += 1

    # Close the last chunk file
    if output_file:
        output_file.close()

    total_chunks = chunk_index - 1
    print(f"Split '{input_file}' into {total_chunks} files, each up to {lines_per_chunk} lines.")


def main():
    parser = argparse.ArgumentParser(
        description="Split a file into fixed-size chunks of lines."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input text file."
    )
    parser.add_argument(
        "-n", "--lines",
        type=int,
        default=600,
        help="Number of lines per chunk (default: 600)."
    )
    parser.add_argument(
        "-o", "--output-prefix",
        help="Prefix for output files (default: based on input file name)."
    )
    args = parser.parse_args()

    split_file(args.input_file, args.lines, args.output_prefix)


if __name__ == "__main__":
    main()
