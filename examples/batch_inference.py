import argparse
import os
from datetime import timedelta
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import jsonlines
from tqdm import tqdm
import torch
from torch import distributed as dist
from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams
# llm=LLM(model="../../../open_models/Qwen2-7B-Instruct/", tensor_parallel_size=4)
from openrlhf.datasets import PromptDataset, SFTDataset, MyPromptDataset
# from vllm import LLM, SamplingParams
# llm=LLM(model="../../../open_models/Qwen2-7B-Instruct/", tensor_parallel_size=4)
from openrlhf.models import Actor, get_llm_for_sequence_regression
# from vllm import LLM, SamplingParams
# llm=LLM(model="../../../open_models/Qwen2-7B-Instruct/", tensor_parallel_size=4)
# here not working
from openrlhf.utils import blending_datasets, get_processor, get_strategy, get_tokenizer
from openrlhf.datasets.utils import load_data
import random
import datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader
seed = 2021
random.seed(seed)
np.random.seed(seed)
import json
from gstop import GenerationStopper

def load_dataset(args, strategy, tokenizer):
    if os.path.exists(os.path.join(args.dataset, 'dataset_dict.json')):
        for idx, split in enumerate(json.load(open(os.path.join(args.dataset, 'dataset_dict.json')))['splits']):
            if idx == 0:
                prompts_data = load_data(os.path.join(args.dataset, split), is_test=False, is_arrow = True)
            else:
                from datasets import concatenate_datasets
                prompts_data = concatenate_datasets([prompts_data, load_data(os.path.join(args.dataset, split), is_test=False, is_arrow = True)])
    else:
        prompts_data = load_data(args.dataset, is_test=False, is_arrow = True)
        if args.dataset2 is not None:
            data2 = load_data(args.dataset2, is_test=False, is_arrow = True, max_samples=args.max_samples)
            from datasets import concatenate_datasets
            prompts_data = concatenate_datasets([prompts_data, data2])
            args.dataset2 = len(data2)
    removed_data = 0
    if args.exist_prompt != 'None':
        exist_prompt_lines = [json.loads(line) for line in open(args.exist_prompt).readlines()]
        exist_prompt = [line['prompt'].lstrip("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>human\n").rstrip("<|im_end|>\n") for line in exist_prompt_lines]
        prompts_data_new = {}
        for item in prompts_data:
            if item['prompt'] not in exist_prompt:
                for k in item.keys():
                    if k not in prompts_data_new.keys():
                        prompts_data_new[k] = []
                    prompts_data_new[k].append(item[k])
            else:
                removed_data += 1
        print('removed_data:', removed_data)

        prompts_data = datasets.Dataset.from_dict(prompts_data_new)
    if args.iter is None:
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    else:
        # for iterative generation
        start_idx = args.iter * args.rollout_batch_size
        end_idx = start_idx + args.rollout_batch_size
        prompts_data = prompts_data.select(range(start_idx, min(end_idx, len(prompts_data))))

    prompts_dataset = MyPromptDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)
    return prompts_dataset, removed_data


def batch_generate_vllm(args):
    from vllm import LLM, SamplingParams

    # configure strategy
    class Empty:
        pass
    os.environ['MASTER_PORT'] = '12755' 
    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args

    # configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)

     # configure model
    print('loading llm.................')
    llm = LLM(model=args.pretrain, tensor_parallel_size=args.tp_size, trust_remote_code=True, seed=args.seed)

    # Create a sampling params object.
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        use_beam_search=False,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        include_stop_str_in_output = True
    )
    prompts_dataset, removed_data = load_dataset(args, strategy=dummy_strategy, tokenizer=tokenizer)
    prompts = prompts_dataset.prompts
    meta_infos = prompts_dataset.meta_info


    # Conditional SFT inference
    if args.enable_ca:
        for i in range(len(prompts)):
            prompts[i] += args.ca_prompt.strip() + " "

    # best of n
    N = args.best_of_n

    def process(prompts, meta_infos, llm, writer):
        for k in range(0, len(prompts), 200):
            outputs = llm.generate(prompts[k:k+200] * N, sampling_params)
            for idx, output in enumerate(outputs):
                prompt = output.prompt
                output = output.outputs[0].text
                result = {"prompt": prompt, "gen": output}
                for k in ['tag', 'test_id', 'chosen']:
                    if k in meta_infos[0].keys():
                        result[k] = meta_infos[idx][k]
                writer.write(result)
    
    print('start generation..........')
    process(prompts, meta_infos, llm, jsonlines.open(args.output_path, mode="w"))

def batch_generate(args):
    from deepspeed.accelerator import get_accelerator
    import gc
    gc.collect()
    get_accelerator().empty_cache()
    # configure strategy
    strategy = get_strategy(args)
    os.environ['MASTER_PORT'] = '12355' 
    strategy.setup_distributed(timeout=timedelta(minutes=120))

    # configure model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # tokenizer
    def tokenize_fn(texts):
        batch = tokenizer(texts, return_tensors="pt", max_length=args.prompt_max_len, padding=True, truncation=True,)
        return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}
    
    prompts_dataset, removed_data = load_dataset(args, strategy, tokenizer=tokenizer)

    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset, args.micro_batch_size, True, False, drop_last=False # sampler ='use_none'??
    )
    pbar = tqdm(
        prompts_dataloader,
        disable=not strategy.is_rank_0(),
    )

    dist.barrier()
    N = args.best_of_n
    
     # prepare models
    model = strategy.prepare(model)
    model.eval()

    stop_tokens = {"<|im_end|>": [151645]}
    stopper = GenerationStopper(stop_tokens)
    
    writer = jsonlines.open(args.output_path + str(strategy.get_rank()), mode="w")
    for prompts, meta_infos in prompts_dataloader:
        # Conditional SFT inference
        output_dataset = []
        if args.enable_ca:
            for i in range(len(prompts)):
                prompts[i] += args.ca_prompt.strip() + " "   
        inputs = tokenize_fn(prompts)   
        for _ in range(N):
            # torch.manual_seed(torch.seed() + _) 
            outputs = model.model.generate(
                **inputs,
                use_cache=True,
                do_sample=not args.greedy_sampling,
                top_p=args.top_p,
                early_stopping=False,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams, 
                # stopping_criteria=stopper.criteria,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            inputs2 = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            idx = 0
            for prompt, output, input_ in zip(prompts, outputs, inputs2):
                output = output[len(input_):]
                if 'Human' in args.input_template and 'Assistant' not in args.input_template:
                    output = 'Assistant:' + output
                output_dataset.append({"prompt": prompt, "gen": output})
                for k in ['tag', 'test_id', 'chosen']:
                    if k in meta_infos.keys():
                        try:
                            output_dataset[-1][k] = meta_infos[k][idx].item()
                        except:
                            output_dataset[-1][k] = meta_infos[k][idx]
                idx += 1
        writer.write_all(output_dataset)
        writer._fp.flush()
        pbar.update()
        dist.barrier()

    # concate multiple output files in rank 0
    if strategy.is_rank_0():
        output_dataset = []
        world_size = dist.get_world_size()
        files = [args.output_path + str(rank) for rank in range(world_size)]
        for file in files:
            with jsonlines.open(file, mode="r") as reader:
                for obj in reader:
                    output_dataset.append(obj)
            os.remove(file)
        jsonlines.open(args.output_path, mode="w").write_all(output_dataset)
        if removed_data > 0:
            jsonlines.open(args.output_path, mode="a").write_all(exist_prompt_lines)

def batch_rm_inference(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=180))

    # configure model
    # load huggingface model/config
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        normalize_reward=True,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # prepare models
    model = strategy.prepare(model)
    model.eval()

    dataset = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        return_eval=False,
        max_count=args.max_samples,
    )
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = SFTDataset(
        dataset, tokenizer, args.max_len, strategy, pretrain_mode=False, input_template=args.input_template
    )
    dataloader = strategy.setup_dataloader(
        dataset, args.micro_batch_size, True, False, dataset.collate_fn, drop_last=False
    )
    pbar = tqdm(
        dataloader,
        disable=not strategy.is_rank_0(),
    )

    dist.barrier()

    output_dataset = []
    with torch.no_grad():
        for _, input_ids, attention_masks, info in pbar:
            input_ids = input_ids.squeeze(1).to(torch.cuda.current_device())
            attention_masks = attention_masks.squeeze(1).to(torch.cuda.current_device())
            rewards = model(input_ids, attention_masks)
            for prompt, output, reward in zip(info["input"], info["output"], rewards):
                output_dataset.append({"input": prompt, "output": output, "reward": reward.item()})

            dist.barrier()

    with jsonlines.open(args.output_path + str(strategy.get_rank()), mode="w") as writer:
        writer.write_all(output_dataset)

    # wait unitl all processes generate done
    dist.barrier()

    # concate multiple output files in rank 0
    if strategy.is_rank_0():
        output_dataset = []
        world_size = dist.get_world_size()
        files = [args.output_path + str(rank) for rank in range(world_size)]
        for file in files:
            with jsonlines.open(file, mode="r") as reader:
                for obj in reader:
                    output_dataset.append(obj)
            # os.remove(file)

        rewards = torch.tensor([obj["reward"] for obj in output_dataset])
        print(f"Reward mean: {rewards.mean().item()}, std: {rewards.std().item()}")

        if args.post_processor and args.post_processor != "null":
            strategy.print(f"Use Processor {args.post_processor}, Reward Norm {args.normalize_reward}")
            processor = get_processor(args.post_processor)
            output_dataset = processor(args, output_dataset)

        with jsonlines.open(args.output_path, mode="w") as writer:
            writer.write_all(output_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_task", type=str, default=None, help="set to generate, generate_vllm or rm")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset2", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=100000000)
    parser.add_argument("--seed", type=int, default=1234)

    # custom dataset key name
    parser.add_argument("--input_key", type=str, default=None)
    parser.add_argument("--output_key", type=str, default=None)

    # for generation
    parser.add_argument("--ta_prompt", type=str, default=None)
    parser.add_argument("--prompt_max_len", type=int, default=4096)
    parser.add_argument("--greedy_sampling", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=float, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--best_of_n", type=int, default=1)
    parser.add_argument("--input_template", type=str, default="Human:\n{}\nAssistant:\n")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--length_penalty", type=float, default=1.5)
    parser.add_argument("--num_beams", type=float, default=3)
    parser.add_argument(
        "--post_processor",
        type=str,
        default=None,
        help="set to rs (Rejection Sampling), ca (Conditional SFT) or None",
    )
    # for vllm
    parser.add_argument("--tp_size", type=int, default=8)

    # for Iterative generation and Rejection Sampling
    parser.add_argument("--iter", type=int, default=None)
    parser.add_argument("--rollout_batch_size", type=int, default=2048)

    # for Conditional SFT
    parser.add_argument("--normalize_reward", action="store_true", default=False)
    parser.add_argument("--reward_template", type=str, default=None)
    parser.add_argument("--enable_ca", action="store_true", default=False)
    parser.add_argument("--ca_prompt", type=str, default="<rm_score>: 5.00", help="Conditional SFT prompt")
    parser.add_argument("--exist_prompt", type=str, default="None",)

    
    args = parser.parse_args()
    if args.eval_task and args.eval_task == "generate":
        batch_generate(args)
    elif args.eval_task and args.eval_task == "generate_vllm":
        batch_generate_vllm(args)
    elif args.eval_task and args.eval_task == "rm":
        batch_rm_inference(args)
    else:
        print("Invalid or missing '--eval_task' argument. Please specify either 'generate' or 'rm'.")
