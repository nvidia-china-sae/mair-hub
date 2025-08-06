import json
from tqdm import tqdm
import re

def load_jsonl(file_path: str):
    """Load data from jsonl file."""
    data = []
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            if item["language"] == "zh":
                # check if there is any english in the text
                if item["duration"] < 30:
                    count += 1
                    item["text"] = item["text"].lower()
                    if re.search(r'[a-z]', item["text"]):
                        print(item["text"])
                        continue
                    else:
                        data.append(item)
            if count > 80000:
                break
    print(f"Total data: {len(data)}")
    return data

if __name__ == "__main__":
    jsonl_file = "data/emilia_zh.jsonl"
    data = load_jsonl(jsonl_file)
    with open(f"./data/{jsonl_file.split('/')[-1].split('.')[0]}-zh-filtered.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")