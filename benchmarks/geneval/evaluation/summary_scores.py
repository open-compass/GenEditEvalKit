# Get results of evaluation

import argparse
import os

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

output_dir = os.path.dirname(args.filename)
file_handler = logging.FileHandler(os.path.join(output_dir, "results.log"), mode='a')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load classnames

with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
    classnames = [line.strip() for line in cls_file]
    cls_to_idx = {"_".join(cls.split()):idx for idx, cls in enumerate(classnames)}

# Load results

df = pd.read_json(args.filename, orient="records", lines=True)

# Measure overall success

logger.info("Summary")
logger.info("=======")
logger.info(f"Total images: {len(df)}")
logger.info(f"Total prompts: {len(df.groupby('metadata'))}")
logger.info(f"% correct images: {df['correct'].mean():.2%}")
logger.info(f"% correct prompts: {df.groupby('metadata')['correct'].any().mean():.2%}")
logger.info("")

# By group

task_scores = []

logger.info("Task breakdown")
logger.info("==============")
for tag, task_df in df.groupby('tag', sort=False):
    task_scores.append(task_df['correct'].mean())
    logger.info(f"{tag:<16} = {task_df['correct'].mean():.2%} ({task_df['correct'].sum()} / {len(task_df)})")
logger.info("")

logger.info(f"Overall score (avg. over tasks): {np.mean(task_scores):.5f}")