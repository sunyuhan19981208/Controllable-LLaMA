import json
import os
import pdb
from datasets import load_dataset
results = {}
for f in os.listdir('qa_full'):
    alpha = float(f[:-5].split('_')[-1])
    with open(f'qa_full/{f}') as fp:
        results[alpha] = json.load(fp)
dataset = load_dataset("json", data_files='/mnt/cachenew/sunyuhan/alpaca-lora/wiki-trivia-qa.json')
data = []
for alpha, result in results.items():
    start = 10000
    val_num = 200
    result = {x['index']:x['output_text'] for x in result}
    pred = [0 for _ in range(4)]
    for i, data_point in enumerate(dataset['train']):
        if i < start:
            continue
        if i >= start + val_num:
            break
        lalel_bit = 1 if data_point['answers'] == 'No relevant information available' else 0
        pred_bit = 2 if 'No relevant information available' in result[i] else 0
        pred[lalel_bit | pred_bit] = pred[lalel_bit | pred_bit] + 1
    if pred[3] > 0:
        recall = pred[3] / (pred[1] + pred[3])
        precision = pred[3] / (pred[3] + pred[2])
    else:
        recall = 0
        precision = 0
    if alpha != 2.0 and alpha >= 0.6:
        data.append((alpha, recall, precision))
    print(f'alpha: {alpha}, pred: {pred}, recall: {recall}, precision: {precision}')
data = sorted(data, key=lambda x: x[0])
x = [d[0] for d in data]
recall = [d[1] for d in data]
precision = [d[2] for d in data]
import matplotlib.pyplot as plt
plt.plot(x, recall, label= 'recall')
plt.plot(x, precision, label= 'precision')

# 设置图表标题和轴标签
plt.title('LLM = LLM + alpha * LoRA')
plt.xlabel('alpha')
plt.ylabel('recall/precision')

x_point = 1.0
prompt_recall = 0.53125
prompt_precision = 0.3148148148148148

# Mark the point with a red dot
plt.scatter(x_point, prompt_recall, color='red', marker='x', label='recall of prompt')
# Mark the point with a red dot
plt.scatter(x_point, prompt_precision, color='blue', marker='x', label='precision of prompt')

plt.legend()

plt.savefig('/mnt/cachenew/sunyuhan/alpaca-lora/refuse_plot.png')
plt.close()

# pdb.set_trace()