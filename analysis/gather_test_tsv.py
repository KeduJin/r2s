"""
This is a script to gather the test tsv files from the output directory
"""

import argparse
import glob
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_output_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    tsv_file_list = glob.glob(os.path.join(args.test_output_path, "*rank*.tsv"))
    if len(tsv_file_list) == 0:
        raise ValueError(
            f"No tsv file found in {args.test_output_path} need to be gathered."
        )
    tsv_file_list.sort(key=lambda x: int(x.split(".")[0][-1]))
    all_content = []
    header = open(tsv_file_list[0], "r").readlines()[0]

    # we record the entry id to avoid duplicate entries
    entry_id_set = set()
    for tsv_file_path in tsv_file_list:
        with open(tsv_file_path, "r") as f:
            lines = f.readlines()[1:]
        for line in lines:
            entry_id = line.split("\t")[0]
            if entry_id in entry_id_set:
                continue
            entry_id_set.add(entry_id)
            all_content.append(line)

    with open(os.path.join(args.test_output_path, "sequence_output.tsv"), "w") as f:
        f.write(header)
        for content in all_content:
            f.write(content)

    print(
        f"Gathered {len(all_content)} lines of content from {len(tsv_file_list)} tsv files."
    )
    # remove the tsv files
    for tsv_file_path in tsv_file_list:
        os.remove(tsv_file_path)


if __name__ == "__main__":
    main()
