import pdb
from typing import final
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from datasets import load_dataset
from torch import multiprocessing, tensor
import torch.distributed as dist
from utils.prompter import Prompter
import multiprocessing as mp
from transformers import GenerationConfig
from fire import Fire
prompt_template_name = 'short'
world_size = int(os.environ.get("WORLD_SIZE", 0))
rank = int(os.environ.get("LOCAL_RANK", 0))
###
def test():
    dataset = load_dataset("json", data_files='/mnt/cachenew/sunyuhan/alpaca-lora/alpaca_data.json')
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    data_li = []
    for i, data_point in enumerate(dataset['train']):
        if i % 8 != rank:
            continue
        if i >= 100:
            break
        # user_prompt = prompter.generate_prompt(
        #     data_point["instruction"], data_point["input"]
        # )
        data_li.append(data_point)
        # print('[SYH] input_ids: ',len(input_ids[0]))
        # res += len(output[0]) - len(input_ids[0])
    dist.barrier()
    gathered_li = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_li, data_li)
    final_li = []
    if rank == 0:
        for li in gathered_li:
            final_li += li
        file_path = "data.json"
        # Write the list of dictionaries to the JSON file
        import json
        with open(file_path, 'w') as json_file:
            json.dump(final_li, json_file)
        print(len(final_li))
# test()
###
base_model = AutoModelForCausalLM.from_pretrained("/mnt/cachenew/sunyuhan/models/Qwen-7B-Chat",
     torch_dtype=torch.float16,trust_remote_code=True).to(rank)
tokenizer = AutoTokenizer.from_pretrained("/mnt/cachenew/sunyuhan/models/Qwen-7B-Chat",trust_remote_code=True)
dataset = load_dataset("json", data_files='/mnt/cachenew/sunyuhan/alpaca-lora/alpaca_data.json')

dist.init_process_group("gloo", rank=rank, world_size=world_size)
def create_dataset():
    prompter = Prompter('short')
    data_li = []
    for i, data_point in enumerate(dataset['train']):
        if i % 8 != rank:
            continue
        if i >= 10000:
            break
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(rank)
        output = base_model.generate(input_ids, max_length=512, temperature=0.7)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True).split('### Response:')[-1]
        print(output_text)
        data_li.append({
            'instruction':data_point["instruction"],
            'input':data_point["input"],
            'output':output_text
        })
    dist.barrier()

    gathered_li = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_li, data_li)
    final_li = []
    if rank == 0:
        for li in gathered_li:
            final_li += li
        file_path = "/mnt/cachenew/sunyuhan/alpaca-lora/qwen/syh_short_alpaca.json"
        # Write the list of dictionaries to the JSON file
        import json
        with open(file_path, 'w') as json_file:
            json.dump(final_li, json_file)
        print(len(final_li))

def create_cot_dataset():
    
    prompter = Prompter('cot')
    data_li = []
    for i, data_point in enumerate(dataset['train']):
        if i % 8 != rank:
            continue
        if i >= 10000:
            break
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(rank)
        output = base_model.generate(input_ids, max_length=512, temperature=0.7)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True).split('### Response:')[-1]
        print(output_text)
        data_li.append({
            'instruction':data_point["instruction"],
            'input':data_point["input"],
            'output':output_text
        })
    dist.barrier()

    gathered_li = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_li, data_li)
    final_li = []
    if rank == 0:
        for li in gathered_li:
            final_li += li
        file_path = "/mnt/cachenew/sunyuhan/alpaca-lora/qwen/syh_cot_alpaca.json"
        # Write the list of dictionaries to the JSON file
        import json
        with open(file_path, 'w') as json_file:
            json.dump(final_li, json_file)
        print(len(final_li))

def create_qa_dataset():
    dataset = load_dataset("json", data_files='/mnt/cachenew/sunyuhan/alpaca-lora/wiki-trivia-qa.json')
    prompter = Prompter('norefqa')
    data_li = []
    refuse_cnt = 0
    for i, data_point in enumerate(dataset['train']):
        if i % 8 != rank:
            continue
        if i >= 10000:
            break
        user_prompt = prompter.generate_qa_prompt(
            data_point["context"], data_point["query"]
        )
        input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(rank)
        output = base_model.generate(input_ids, max_length=512, temperature=0.7)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True).split('### Response:')[-1]
        print(i)

        data_li.append({
            'context':data_point["context"],
            'query':data_point["query"],
            'output':output_text
        })
        if 'No relevant information available' in output_text:
            refuse_cnt += 1
    dist.barrier()

    gathered_li = [None for _ in range(world_size)]
    cnt_li = [None for _ in range(world_size)]
    dist.all_gather_object(cnt_li, refuse_cnt)
    dist.all_gather_object(gathered_li, data_li)
    
    final_li = []
    if rank == 0:
        print('Refuse Count:', sum(cnt_li))
        for li in gathered_li:
            final_li += li
        file_path = "/mnt/cachenew/sunyuhan/alpaca-lora/qwen/wiki-trivia-qa-noref.json"
        # Write the list of dictionaries to the JSON file
        import json
        with open(file_path, 'w') as json_file:
            json.dump(final_li, json_file)
        print(len(final_li))

def create_short_qa_dataset():
    dataset = load_dataset("json", data_files='/mnt/cachenew/sunyuhan/alpaca-lora/wiki-trivia-qa.json')
    prompter = Prompter('shortqa')
    data_li = []
    for i, data_point in enumerate(dataset['train']):
        if i % 8 != rank:
            continue
        if i >= 10000:
            break
        user_prompt = prompter.generate_qa_prompt(
            data_point["context"], data_point["query"]
        )
        input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(rank)
        output = base_model.generate(input_ids, max_length=512, temperature=0.7)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True).split('### Response:')[-1]
        print(i)
        data_li.append({
            'context':data_point["context"],
            'query':data_point["query"],
            'output':output_text
        })
    dist.barrier()

    gathered_li = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_li, data_li)
    
    final_li = []
    if rank == 0:
        for li in gathered_li:
            final_li += li
        file_path = "/mnt/cachenew/sunyuhan/alpaca-lora/wiki-trivia-qa-short.json"
        # Write the list of dictionaries to the JSON file
        import json
        with open(file_path, 'w') as json_file:
            json.dump(final_li, json_file)
        print(len(final_li))

def qa():
    dataset = load_dataset("json", data_files='/mnt/cachenew/sunyuhan/alpaca-lora/wiki-trivia-qa.json')
    prompter = Prompter('qa')
    from peft import PeftModel
    lora_weights = '/mnt/cachenew/sunyuhan/alpaca-lora/qwen/qa-231126-10000'
    global base_model, tokenizer
    results = []
    for alpha in range(51):
        alpha = alpha / 50
        model = PeftModel.from_pretrained(
            base_model,
            lora_weights,
            torch_dtype=torch.float16,
        )
        if alpha != 1:
            for k in [k for k in model.state_dict().keys() if 'lora_A' in k]:
                model.state_dict()[k] *= alpha
        # pdb.set_trace()
        refuse_cnt = 0
        start = 10000
        val_num = 200
        for i, data_point in enumerate(dataset['train']):
            if i % 8 != rank or i < start:
                continue
            if i >= start + val_num:
                break
            user_prompt = prompter.generate_qa_prompt(
                data_point["context"], data_point["query"]
            )
            input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(rank)
            # output = model.generate(input_ids, max_length=512, temperature=0.7)
            generation_config = GenerationConfig(
                temperature=0.7
            )
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=512,
                )
            s = generation_output.sequences[0]
            output_text = tokenizer.decode(s, skip_special_tokens=True).split('### Response:')[-1]
            # print(output_text)
            if 'No relevant information available' in output_text:
                refuse_cnt += 1
        dist.barrier()
        gathered_li = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_li, refuse_cnt)
        if rank == 0:
            print(sum(gathered_li))
            results.append({
                'alpha': alpha,
                'refuse_rate': sum(gathered_li) / val_num
            })
    file_path = "/mnt/cachenew/sunyuhan/alpaca-lora/qwen/qa_exp1127.json"
    # Write the list of dictionaries to the JSON file
    import json
    with open(file_path, 'w') as json_file:
        json.dump(results, json_file)

def prompt_short():
    prompter = Prompter('short')
    from peft import PeftModel
    lora_weights = '/mnt/cachenew/sunyuhan/alpaca-lora/qwen/short-231121-10000'
    global base_model, tokenizer
    results = []
    for alpha in [0]:
        alpha = alpha / 100
        model = PeftModel.from_pretrained(
            base_model,
            lora_weights,
            torch_dtype=torch.float16,
        )
        if alpha != 1:
            for k in [k for k in model.state_dict().keys() if 'lora_A' in k]:
                model.state_dict()[k] *= alpha
        # pdb.set_trace()
        total_len = 0
        start = 10000
        val_num = 100
        for i, data_point in enumerate(dataset['train']):
            if i % 8 != rank or i < start:
                continue
            if i >= start + val_num:
                break
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(rank)
            # output = model.generate(input_ids, max_length=512, temperature=0.7)
            generation_config = GenerationConfig(
                temperature=0.7
            )
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=512,
                )
            s = generation_output.sequences[0]
            total_len += len(s) - len(input_ids[0])
        dist.barrier()
        gathered_li = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_li, total_len)
        if rank == 0:
            print(sum(gathered_li))
def prompt_refuse():
    dataset = load_dataset("json", data_files='/mnt/cachenew/sunyuhan/alpaca-lora/wiki-trivia-qa.json')
    prompter = Prompter('norefqa')
    from peft import PeftModel
    lora_weights = '/mnt/cachenew/sunyuhan/alpaca-lora/qa-noref-231007-10000'
    global base_model, tokenizer
    results = []
    for alpha in [0]:
        alpha = alpha / 50
        model = PeftModel.from_pretrained(
            base_model,
            lora_weights,
            torch_dtype=torch.float16,
        )
        if alpha != 1:
            for k in [k for k in model.state_dict().keys() if 'lora_A' in k]:
                model.state_dict()[k] *= alpha
        # pdb.set_trace()
        refuse_cnt = 0
        start = 10000
        val_num = 200
        for i, data_point in enumerate(dataset['train']):
            if i % 8 != rank or i < start:
                continue
            if i >= start + val_num:
                break
            user_prompt = prompter.generate_qa_prompt(
                data_point["context"], data_point["query"]
            )
            input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(rank)
            # output = model.generate(input_ids, max_length=512, temperature=0.7)
            generation_config = GenerationConfig(
                temperature=0.7
            )
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=512,
                )
            s = generation_output.sequences[0]
            output_text = tokenizer.decode(s, skip_special_tokens=True).split('### Response:')[-1]
            # print(output_text)
            if 'No relevant information available' in output_text:
                refuse_cnt += 1
        dist.barrier()
        gathered_li = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_li, refuse_cnt)
        if rank == 0:
            print(sum(gathered_li) / val_num)

def lora():
    prompter = Prompter('alpaca')
    from peft import PeftModel
    lora_weights = '/mnt/cachenew/sunyuhan/alpaca-lora/qwen/short-231121-10000'
    global base_model, tokenizer
    results = []
    for alpha in range(0, 101):
        alpha = alpha / 100
        model = PeftModel.from_pretrained(
            base_model,
            lora_weights,
            torch_dtype=torch.float16,
        )
        if alpha != 1:
            for k in [k for k in model.state_dict().keys() if 'lora_A' in k]:
                model.state_dict()[k] *= alpha
        # pdb.set_trace()
        total_len = 0
        start = 10000
        val_num = 100
        for i, data_point in enumerate(dataset['train']):
            if i % 8 != rank or i < start:
                continue
            if i >= start + val_num:
                break
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(rank)
            # output = model.generate(input_ids, max_length=512, temperature=0.7)
            generation_config = GenerationConfig(
                temperature=0.7
            )
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=512,
                )
            s = generation_output.sequences[0]
            total_len += len(s) - len(input_ids[0])
        dist.barrier()
        gathered_li = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_li, total_len)
        if rank == 0:
            print(sum(gathered_li))
            results.append({
                'alpha': alpha,
                'output_len': sum(gathered_li)
            })
    file_path = "/mnt/cachenew/sunyuhan/alpaca-lora/qwen/alpha_exp1122.json"
    # Write the list of dictionaries to the JSON file
    import json
    with open(file_path, 'w') as json_file:
        json.dump(results, json_file)

def refuse_short_qa():
    dataset = load_dataset("json", data_files='/mnt/cachenew/sunyuhan/alpaca-lora/wiki-trivia-qa.json')
    prompter = Prompter('qa')
    from peft import PeftModel
    loras = ['/mnt/cachenew/sunyuhan/alpaca-lora/qa-noref-231007-10000', '/mnt/cachenew/sunyuhan/alpaca-lora/lora-short-231004-10000']
    # loras = ['/mnt/cachenew/sunyuhan/alpaca-lora/qa-short-231007-10000']
    global base_model, tokenizer
    results = []
    # loras.reverse()
    alpha_list = []
    for i in range(11):
        for j in range(11):
            alpha_list.append((i/10,j/10))
    for alphas in alpha_list:
        import copy
        model = copy.deepcopy(base_model)
        model.to(rank)
        result = {}
        for lora_weights,alpha in zip(loras, alphas):
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
            if 'noref' in lora_weights:
                result['noref'] = alpha
            if 'short' in lora_weights:
                result['short'] = alpha
            if alpha != 1:
                for k in [k for k in model.state_dict().keys() if 'lora_A' in k]:
                    model.state_dict()[k] *= alpha
            # pdb.set_trace()
            model = model.merge_and_unload()
            # pdb.set_trace()
        # pdb.set_trace()
        total_len = 0
        refuse_cnt = 0
        start = 10000
        val_num = 200
        batch_size = val_num // world_size
        data = dataset['train'][start+rank*batch_size:start+(rank+1)*batch_size]
        input = []
        for i in range(batch_size):
            user_prompt = prompter.generate_qa_prompt(
                data["context"][i], data["query"][i]
            )
            input.append(user_prompt)
        tokenizer.pad_token = tokenizer.eos_token
        input_ids = tokenizer(input, padding='longest', return_tensors='pt').input_ids.to(rank)
        generation_config = GenerationConfig(
                temperature=0.7
            )
        mini_batch_size = 8
        mini_batches = [input_ids[i:i+mini_batch_size] for i in range(0, len(input_ids), mini_batch_size)]
        for mini_batch in mini_batches:
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=mini_batch,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=512,
                )
            s = generation_output.sequences
            pdb.set_trace()
        for i, data_point in enumerate(dataset['train']):
            if i % 8 != rank or i < start:
                continue
            if i >= start + val_num:
                break
            user_prompt = prompter.generate_qa_prompt(
                data_point["context"], data_point["query"]
            )
            input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(rank)
            # output = model.generate(input_ids, max_length=512, temperature=0.7)
            generation_config = GenerationConfig(
                temperature=0.7
            )
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=512,
                )
            s = generation_output.sequences[0]
            total_len += len(s) - len(input_ids[0])
            output_text = tokenizer.decode(s, skip_special_tokens=True).split('### Response:')[-1]
            # print(output_text)
            if 'No relevant information available' in output_text:
                refuse_cnt += 1
        dist.barrier()
        gathered_li = [None for _ in range(world_size)]
        gathered_len = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_li, refuse_cnt)
        dist.all_gather_object(gathered_len, total_len)
        if rank == 0:
            result['refuse_cnt'] = sum(gathered_li)
            result['total_len'] = sum(gathered_len)
            print(result)
            results.append(result)
    if rank == 0:
        file_path = "/mnt/cachenew/sunyuhan/alpaca-lora/qa_exp100901.json"
        # Write the list of dictionaries to the JSON file
        import json
        with open(file_path, 'w') as json_file:
            json.dump(results, json_file)

def short_full_response():
    prompter = Prompter('alpaca')
    from peft import PeftModel
    lora_weights = '/mnt/cachenew/sunyuhan/alpaca-lora/lora-short-231004-10000'
    global base_model, tokenizer
    results = []
    for alpha in range(11):
        alpha = alpha / 10
        model = PeftModel.from_pretrained(
            base_model,
            lora_weights,
            torch_dtype=torch.float16,
        )
        if alpha != 1:
            for k in [k for k in model.state_dict().keys() if 'lora_A' in k]:
                model.state_dict()[k] *= alpha
        # pdb.set_trace()
        total_len = 0
        responses = []
        start = 10000
        val_num = 100
        for i, data_point in enumerate(dataset['train']):
            if i % 8 != rank or i < start:
                continue
            if i >= start + val_num:
                break
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(rank)
            # output = model.generate(input_ids, max_length=512, temperature=0.7)
            generation_config = GenerationConfig(
                temperature=0.7
            )
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=512,
                )
            s = generation_output.sequences[0]
            output_text = tokenizer.decode(s, skip_special_tokens=True).split('### Response:')[-1]
            responses.append(dict(
                index = i,
                output_text= output_text
            ))
            total_len += len(s) - len(input_ids[0])
        dist.barrier()
        gathered_li = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_li, total_len)
        responses_li = [None for _ in range(world_size)]
        dist.all_gather_object(responses_li, responses)
        if rank == 0:
            output_len = sum(gathered_li)
            print('output_len',output_len)
            results = []
            for r in responses_li:
                results += r
            file_path = f"/mnt/cachenew/sunyuhan/alpaca-lora/short_full_response_{alpha}.json"
            # Write the list of dictionaries to the JSON file
            import json
            with open(file_path, 'w') as json_file:
                json.dump(results, json_file)

def alignment_full_response():
    prompter = Prompter('alignment')
    global base_model, tokenizer
    results = []
    model = base_model
    total_len = 0
    responses = []
    start = 10000
    val_num = 100
    for i, data_point in enumerate(dataset['train']):
        if i % 8 != rank or i < start:
            continue
        if i >= start + val_num:
            break
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(rank)
        # output = model.generate(input_ids, max_length=512, temperature=0.7)
        generation_config = GenerationConfig(
            temperature=0.7
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=512,
            )
        s = generation_output.sequences[0]
        output_text = tokenizer.decode(s, skip_special_tokens=True).split('### Response:')[-1]
        print(output_text)
        responses.append(dict(
            index = i,
            output_text= output_text
        ))
        total_len += len(s) - len(input_ids[0])
    dist.barrier()
    gathered_li = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_li, total_len)
    responses_li = [None for _ in range(world_size)]
    dist.all_gather_object(responses_li, responses)
    if rank == 0:
        output_len = sum(gathered_li)
        print('output_len',output_len)
        results = []
        for r in responses_li:
            results += r
        file_path = f"/mnt/cachenew/sunyuhan/alpaca-lora/alignment_full_response.json"
        # Write the list of dictionaries to the JSON file
        import json
        with open(file_path, 'w') as json_file:
            json.dump(results, json_file)

def qa_full_response():
    dataset = load_dataset("json", data_files='/mnt/cachenew/sunyuhan/alpaca-lora/wiki-trivia-qa.json')
    prompter = Prompter('norefqa')
    from peft import PeftModel
    lora_weights = '/mnt/cachenew/sunyuhan/alpaca-lora/qa-noref-231007-10000'
    global base_model, tokenizer
    results = []
    for alpha in [0]:
        alpha = alpha / 10
        model = PeftModel.from_pretrained(
            base_model,
            lora_weights,
            torch_dtype=torch.float16,
        )
        if alpha != 1:
            for k in [k for k in model.state_dict().keys() if 'lora_A' in k]:
                model.state_dict()[k] *= alpha
        # pdb.set_trace()
        total_len = 0
        responses = []
        start = 10000
        val_num = 200
        for i, data_point in enumerate(dataset['train']):
            if i % 8 != rank or i < start:
                continue
            if i >= start + val_num:
                break
            user_prompt = prompter.generate_qa_prompt(
                data_point["context"], data_point["query"]
            )
            input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(rank)
            # output = model.generate(input_ids, max_length=512, temperature=0.7)
            generation_config = GenerationConfig(
                temperature=0.7
            )
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=512,
                )
            s = generation_output.sequences[0]
            output_text = tokenizer.decode(s, skip_special_tokens=True).split('### Response:')[-1]
            responses.append(dict(
                index = i,
                output_text= output_text
            ))
            total_len += len(s) - len(input_ids[0])
        dist.barrier()
        gathered_li = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_li, total_len)
        responses_li = [None for _ in range(world_size)]
        dist.all_gather_object(responses_li, responses)
        if rank == 0:
            output_len = sum(gathered_li)
            print('output_len',output_len)
            results = []
            for r in responses_li:
                results += r
            file_path = f"/mnt/cachenew/sunyuhan/alpaca-lora/qa_full/qa_full_response_{alpha}_prompt.json"
            # Write the list of dictionaries to the JSON file
            import json
            with open(file_path, 'w') as json_file:
                json.dump(results, json_file)

if __name__ == "__main__":
    Fire()