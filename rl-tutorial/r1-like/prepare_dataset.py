from datasets import load_dataset
import json
from transformers import AutoTokenizer


user_prompt = """Please reason step by step, and put your final answer within \\boxed{}."""


tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        

def download_and_save_dataset(dataset, split="train", output_file="dataset.json"):

    data_list = []
    for item in dataset:
        question = ""
        answer = ""

        for i in item:
            if i["from"] =="human" or i["from"] == "Human":
                question = i["value"]
            elif i["from"] == "assistant" or i["from"] == "Assistant":
                answer = i["ground_truth"]["value"]

        # 使用 Qwen 的对话模板格式化问答对
        # 构建对话格式
        if question != "" and answer != "":
            messages = [
                {"role": "user", "content": question + " " + user_prompt},
            ]
        else:
            continue
        
        # 应用 Qwen 的对话模板
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # print("formatted_prompt: ", formatted_prompt)
        # 将格式化后的对话添加到数据列表
        data_list.append({
            "input": formatted_prompt,
            "ground_truth_answer": answer,
            "problem": question,
        })
    # 将数据保存为 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    
    print(f"数据集已保存至 {output_file}")

if __name__ == "__main__":

    dataset_name = "./orz_math_57k_collected.json"
    with open(dataset_name, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # print(data_list[0:5])
    download_and_save_dataset(data_list, split="train", output_file="orz-math-57k-distill.json")
