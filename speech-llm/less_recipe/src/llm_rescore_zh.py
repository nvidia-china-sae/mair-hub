# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import csv
import math
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional, List
from openai import OpenAI
from tqdm import tqdm
import editdistance as ed

API_URL = "https://integrate.api.nvidia.com/v1"
API_KEY = "YOUR_API_KEY"
LLM_MODEL = "01-ai/yi-large" #YOUR_MODEL_NAME


WORLD_SIZE = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
RANK = int(os.environ['RANK']) if 'RANK' in os.environ else 1


@dataclass
class Role:
    ASSISTANT = 'assistant'
    SYSTEM = 'system'
    USER = 'user'


def is_en(word):
    for c in word:
        if ord(c) in range(97,123) or ord(c) in range(65,91):
            return True
    return False

def is_zh(word):
    #print(word)
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def is_enchar(ch):
    if ord(ch) in range(97,122) or ord(ch) in range(65,90):
        return True
    else:
        return False

def is_zhchar(ch):
    #print(word)
    if '\u4e00' <= ch <= '\u9fff':
        return True
    else:
        return False

def read_hypo_and_wer(hypo_manifest: str, hypo_key: str, wer_key: Optional[str] = None, return_json=False): 
    hypo_list, wer_list = [], []
    groundtruth_list = []
    json_dict_list = []
    f = open(hypo_manifest, 'r').readlines()[RANK-1::WORLD_SIZE]
    for line in tqdm(f, total=len(f)):
        try:
            json_line = json.loads(line)
            hypo_list.append(json_line["supervisions"][0]["text"].strip())
            wer_list.append(json_line[wer_key] if wer_key else None)
            groundtruth_list.append(json_line["supervisions"][0]["text"])
            json_dict_list.append(json_line)
        except Exception as e:
            line = line.strip('\n')
            print(f"Error: {e} in reading line: {line}")
            continue

    if not return_json:
        if wer_key:
            return groundtruth_list, hypo_list, wer_list
        else:
            del wer_list
            return groundtruth_list, hypo_list, None
    else:
        if wer_key:
            return groundtruth_list, hypo_list, wer_list, json_dict_list
        else:
            del wer_list
            return groundtruth_list, hypo_list, None, json_dict_list
        

def history_to_messages(history, system: str):
    messages = [{'role': Role.SYSTEM, 'content': system}]
    for h in history:
        messages.append({'role': Role.USER, 'content': h[0]})
        messages.append({'role': Role.ASSISTANT, 'content': h[1]})
    return messages

def messages_to_history(messages):
    assert messages[0]['role'] == Role.SYSTEM
    system = messages[0]['content']
    history = []
    for q, r in zip(messages[1::2], messages[2::2]):
        history.append([q['content'], r['content']])
    return system, history

def create_llm_client():
    client = OpenAI(api_key=API_KEY, base_url=API_URL)
    # only need to pass message
    ready_client = partial(client.chat.completions.create, model=LLM_MODEL)
    return ready_client

def get_system_message(language_tag):
    assert language_tag in ['zh', 'en'], f"language_tag should be zh or en, but got {language_tag}"
    if language_tag == 'zh':
        system_prompt = "请确认我的需求，并作为语音识别算法专家回复我。我会提供一批语音识别模型的解码结果, 每个句子之间用#隔开, 你将协助我对可能出现的替换、插入、删除错误进行查找和修改，并计算原始结果和修正后结果的错误率。如果是中文，计算字错误率；如果是英文，计算单词错误率。最终输出格式为<修正后结果|错误率>，示例如下：输入：#一学生因初烟被处罚款四零零元#许多够防者缩在大厅里等后摇好买房#。输出：<一学生因抽烟被处罚款四零零元|7.14%>#<许多购房者缩在大厅里等候摇号买房|25.00%>。每次我输入之后，请直接返回结果，不要给出计算过程谢谢。"
    else:
        system_prompt = "Please confirm my requirement and reply to me as an expert in speech recognition algorithms. I will provide a batch of decoding results of ASR model, each sentence is separated by #, and you will assist me to find and correct possible substitution, insertion and deletion errors, and calculate the error rate of the original result and the corrected result. If it is Chinese, calculate the word error rate; if it is English, calculate the word error rate. The final output format is <corrected result|error rate>, the example is as follows: Input: # Nice to meat you # hello word #. Output: <Nice to meet you | 25.00%> #<hello world | 50.00%>. After each time I intput, please return the result directly without giving the calculation process."
    return {"role": Role.SYSTEM, "content": system_prompt}

def get_user_message(hypos: List[str]):
    prompt =f'#{"#".join(hypos)}#'

    return {"role": Role.USER, "content": prompt}

def LLM_post_process(content: str, round: int, i: int, total_len: int, give_up=False):
    """process batch llm predictions and return llm_hypos and wers

    Args:
        content (str): llm batch predictions
        round (int): for debugging, which round of predictions

    Returns:
        llm_hypos: List[str]
        wers: List[float]
    """
    content_list = content.strip().split('#')

    content_list = [line for line in content_list if line.strip()!=""]
    
    if len(content_list) != total_len:
        raise ValueError(f"round: {round} {i}-th, total_len: {total_len}, content_list: {len(content_list)}")
    
    llm_hypos, wers = [], []
    total_process, success_porcess, error_process= 0, 0, 0
    for utt_idx, each_prediction in enumerate(content_list):
        total_process += 1

        if give_up:
            try:
                llm_hypo = each_prediction.split("|")[0]
                llm_hypo = llm_hypo.strip()
                llm_hypos.append(llm_hypo)
                wers.append("-1")
                success_porcess += 1
            except Exception as e:
                print(f'Error: {e} in round: {round}, utt_idx: {utt_idx}, llm_prediction: {each_prediction}')
                llm_hypos.append(f"#ERROR: {each_prediction}#")
                wers.append("-1")
                error_process += 1
                continue
        else:
            llm_hypo = each_prediction
            wer = "-1"
            success_porcess += 1


            llm_hypos.append(llm_hypo)
            wers.append(float(wer))

    if give_up:
        print(f"round: {round} {i}-th, total_process: {total_process}, success_process: {success_porcess}, encounter error: {error_process}")
    else:
        print(f"round: {round} {i}-th, total_process: {total_process}, success_process: {success_porcess}")

    return llm_hypos, wers


class GPT4Model:
    def __init__(self, api_key, url, model_name= "gpt-4o-2024-05-13"):
        self.api_key = api_key
        self.url = url
        self.model_name = model_name

        self.client = OpenAI(api_key=self.api_key, base_url=self.url)
        
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    
    def get_pred(self, messages: List[dict], round: int, total_len: int):
        LLM_hypos, LLM_pred_wers = "", ""
        i, MAX_RETRY = 0, 3
        while i < MAX_RETRY:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages
                )

                content = response.choices[0].message.content
                log_prob = response.choices[0].logprobs

                x, y, z = response.usage.completion_tokens, response.usage.prompt_tokens, response.usage.total_tokens
                self.completion_tokens += x
                self.prompt_tokens += y
                self.total_tokens += z
                print(f"prompt_tokens: {x}, completion_tokens: {y}, total_tokens: {z}")
                
                if i == MAX_RETRY - 1:
                    LLM_hypos, LLM_pred_wers = LLM_post_process(content, round, i=i, total_len=total_len, give_up=True)
                else:
                    LLM_hypos, LLM_pred_wers = LLM_post_process(content, round, i=i, total_len=total_len, give_up=False)
            except ValueError as e:
                i += 1
                time.sleep(1+i / 10)
                print(f'Error: {e} Round: {round} retries {i} times')
            except IndexError as e:
                i += 1
                time.sleep(1+i / 10)
                print(f'Error: {e} Round: {round} retries {i} times')
            else:
                break
            
        return LLM_hypos, LLM_pred_wers
    
    def text_normalize(self, texts: List[str]):
        """
            Return List of List[str(zh-char or en-word)]
            Another return value: List[str(sentence)]
        """
        norm_texts = []
        norm_str_texts = []
        for text in texts:
            if re.search(r'[^\w\s\']', text):
                re.sub(r'[^\w\s\']', "", text)
            norm_text, norm_text_str = [], ""
            en_word = ""
            for char in text:
                if is_zhchar(char):
                    norm_text.append(char)
                    norm_text_str += f" {char}"
                if is_zhchar(char) or re.search(r'\s', char):
                    if en_word:
                        norm_text.append(en_word)
                        norm_text_str += f" {en_word}"
                        en_word = ""
                elif is_enchar(char) or re.search(r'\'', char):
                    en_word += char
            if en_word:
                norm_text.append(en_word)
                norm_text_str += f" {en_word}"
            norm_texts.append(norm_text)
            norm_str_texts.append(norm_text_str.strip())

        return norm_texts, norm_str_texts

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("cmd error, should be: python xxx.py input.json output.json language_tag")
        sys.exit(1)
    
    input_json = sys.argv[1]
    output_json = sys.argv[2]
    language_tag = sys.argv[3]
    
    dataset_tag = os.path.basename(input_json).split('.')[0]
    groundtruths, hypos, greedy_wers, input_json_list = read_hypo_and_wer(input_json, hypo_key='text', wer_key=None, return_json=True)
    LLM_GPT4 = GPT4Model(model_name=LLM_MODEL, api_key=API_KEY, url=API_URL)
    system_message = get_system_message(language_tag=language_tag)
 
    llm_hypos, llm_asr_wers = [], []
    rets = []
    with open(output_json, 'w') as wf:
        start_utts, total_utts = 0, len(groundtruths)
        hypos = hypos[start_utts:total_utts]
        groundtruths = groundtruths[start_utts:total_utts]
        utts_per_round = 5 

        rounds = math.ceil(len(hypos) // utts_per_round)
        for r in tqdm(range(rounds), total=rounds):
            try:
                if r == rounds - 1:
                    round_hypos = hypos[r*utts_per_round:]
                    round_input_json_list = input_json_list[r*utts_per_round:]
                else:
                    round_hypos = hypos[r*utts_per_round:(r+1)*utts_per_round]
                    round_input_json_list = input_json_list[r*utts_per_round:(r+1)*utts_per_round]
            
                user_message = get_user_message(round_hypos)
                batch_messages = [system_message, user_message]
                llm_hypos, llm_pred_wers = LLM_GPT4.get_pred(messages=batch_messages, round=r, total_len=len(round_hypos))

                print(f"user: {user_message}")
                print(f'llm_hypos: {llm_hypos}')
                
                
                list_of_llm_hypo_list, list_of_llm_hypo_str = LLM_GPT4.text_normalize(llm_hypos)
                list_of_hypo_list, list_of_hypo_str = LLM_GPT4.text_normalize(round_hypos)

                if len(round_hypos) == len(llm_hypos):
                    print(f"round_hypos: {len(round_hypos)}")
                    print(f"llm_hypos: {len(llm_hypos)}")
                    
                for idx, (hypo, llm_hypo, llm_pred_wer) in enumerate(
                    zip(list_of_hypo_list, list_of_llm_hypo_list, llm_pred_wers)
                ):
                    gdy_hypo_list = hypo
                    llm_hypo_list = llm_hypo

                    if len(gdy_hypo_list) == 0:
                        hypo_wer = 1.0
                    else:
                        hypo_wer = ed.eval(gdy_hypo_list, llm_hypo_list) / len(gdy_hypo_list)

                    supervision = round_input_json_list[idx]["supervisions"][0]
                    supervision['greedy_text'] = supervision['text']
                    supervision['text'] = list_of_llm_hypo_str[idx]
                    supervision['hypo_wer'] = round(hypo_wer, 4)
                    round_input_json_list[idx]["supervisions"] = [supervision]
                    wf.write(json.dumps(round_input_json_list[idx], ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Error: {e} in round: {r}")
                continue
 
