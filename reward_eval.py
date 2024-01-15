import pdb
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import os
import json
reward_name = "/mnt/cachenew/sunyuhan/models/reward-model-deberta-v3-large-v2"
rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
rank_model = rank_model.to(0)
# question = "I just came out of from jail, any suggestion of my future?"
# helpful = "It's great to hear that you have been released from jail."
# bad = "Go back to jail you scum"

# inputs = tokenizer(question, helpful, return_tensors='pt')
# good_score = rank_model(**inputs).logits[0].cpu().detach()

# inputs = tokenizer(question, bad, return_tensors='pt')
# bad_score = rank_model(**inputs).logits[0].cpu().detach()
def qa():
    dataset = load_dataset("json", data_files='/mnt/cachenew/sunyuhan/alpaca-lora/wiki-trivia-qa.json')
    results = {}
    for f in os.listdir('qa_full'):
        alpha = float(f[:-5].split('_')[-1])
        with open(f'qa_full/{f}') as fp:
            results[alpha] = json.load(fp)
    start = 10000
    val_num = 200
    scores = {a:[] for a in results.keys()}

    for alpha,result in results.items():
        result = {x['index']:x['output_text'] for x in result}
        for i, data_point in enumerate(dataset['train']):
            if i < start:
                continue
            if i >= start + val_num:
                break
            question = data_point['query']
            inputs = tokenizer(question, result[i], return_tensors='pt').to(0)
            score = rank_model(**inputs).logits[0].cpu().detach()
            scores[alpha].append(score)
    def rank_list(input_list):
        # 使用sorted函数对输入列表进行排序，并保留原始元素的索引
        sorted_list = sorted(input_list, key=lambda x: x[1])
        # 创建一个字典，将元素和其排名关联起来
        rank_dict = {element[0]: rank + 1 for rank, element in enumerate(sorted_list)}
        return rank_dict
    rank_results = {a:[] for a in results.keys()}
    for i in range(len(scores[0])):
        tmp_scores = [(k, x[i]) for k, x in scores.items()]
        rank = rank_list(tmp_scores)
        for k, r in rank.items():
            rank_results[k].append(r)
    res = sorted([(k,sum(v)) for k,v in rank_results.items()], key=lambda x: x[0])
    pdb.set_trace()

def alignment():
    dataset = load_dataset("json", data_files='/mnt/cachenew/sunyuhan/alpaca-lora/alpaca_data.json')
    results = {}
    with open(f'/mnt/cachenew/sunyuhan/alpaca-lora/alignment_full_response.json') as fp:
        results[2.0] = json.load(fp)
    with open(f'/mnt/cachenew/sunyuhan/alpaca-lora/short_full/short_full_response_0.0.json') as fp:
        results[0.0] = json.load(fp)
    start = 10000
    val_num = 100
    scores = {a:[] for a in results.keys()}

    for alpha,result in results.items():
        result = {x['index']:x['output_text'] for x in result}
        for i, data_point in enumerate(dataset['train']):
            if i < start:
                continue
            if i >= start + val_num:
                break
            question = data_point['instruction'] + data_point['input']
            inputs = tokenizer(question, result[i], return_tensors='pt').to(0)
            score = rank_model(**inputs).logits[0].cpu().detach()
            scores[alpha].append(score)
    def rank_list(input_list):
        # 使用sorted函数对输入列表进行排序，并保留原始元素的索引
        sorted_list = sorted(input_list, key=lambda x: x[1])
        # 创建一个字典，将元素和其排名关联起来
        rank_dict = {element[0]: rank + 1 for rank, element in enumerate(sorted_list)}
        return rank_dict
    rank_results = {a:[] for a in results.keys()}
    for i in range(len(scores[0])):
        tmp_scores = [(k, x[i]) for k, x in scores.items()]
        rank = rank_list(tmp_scores)
        for k, r in rank.items():
            rank_results[k].append(r)
    res = sorted([(k,sum(v)) for k,v in rank_results.items()], key=lambda x: x[0])
    pdb.set_trace()
alignment()