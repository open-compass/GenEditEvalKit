import pandas as pd
import ast
from collections import defaultdict


result_csv_path = "./results/flux_output.csv"

df = pd.read_csv(result_csv_path)

big_class_stats = defaultdict(lambda: [0, 0])   # 大类： [correct, total]
small_class_stats = defaultdict(lambda: [0, 0]) # 小类： [correct, total]

for _, row in df.iterrows():
    checkpoints = ast.literal_eval(row['testpoint'])
    try:
        scores = ast.literal_eval(row['result_json'])['score'] if isinstance(row['result_json'], str) else row['score']
        
        if not isinstance(scores, list):
            scores = ast.literal_eval(row['score'])

        for cp, score in zip(checkpoints, scores):

            if '-' in cp:
                big_class, small_class = cp.split('-', 1)[0], cp
            else:
                big_class = small_class = cp

            big_class_stats[big_class][1] += 1
            small_class_stats[small_class][1] += 1
            if score == 1:
                big_class_stats[big_class][0] += 1
                small_class_stats[small_class][0] += 1
    except:
        continue


print("📘 Primary Dimension Evaluation Results:")
for big_class, (correct, total) in big_class_stats.items():
    acc = correct / total if total > 0 else 0
    print(f"  - {big_class}: {correct}/{total} = {acc:.2%}")

print("\n📗 Sub Dimension Evaluation Results:")
for small_class in sorted(small_class_stats.keys()):
    correct, total = small_class_stats[small_class]
    acc = correct / total if total > 0 else 0
    print(f"  - {small_class}: {correct}/{total} = {acc:.2%}")
