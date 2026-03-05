import json
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("result_file_prefix", type=str)
args = parser.parse_args()


result_files = glob.glob(args.result_file_prefix + ".*")
results = []
for result_file in result_files:
    with open(result_file, 'r') as f:
        res = f.readlines()
        results.extend(res)

output_file = args.result_file_prefix
with open(output_file, 'w') as f:
    res_ = results[0].rstrip()
    f.write(res_)
    for res in results[1:]:
        f.write("\n")
        res_ = res.rstrip()
        f.write(res_)