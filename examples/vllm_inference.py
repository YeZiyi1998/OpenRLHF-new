import argparse
import os
from datetime import timedelta
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import jsonlines
import torch
from torch import distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import concatenate_datasets
from openrlhf.datasets import PromptDataset, SFTDataset, MyPromptDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.utils import blending_datasets, get_processor, get_strategy, get_tokenizer
from openrlhf.datasets.utils import load_data
import random
import numpy as np
import multiprocessing
seed = 2021
random.seed(seed)
np.random.seed(seed)
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def predict(data, port, args=None, model_name=None,meta_info=None):
    from openai import OpenAI
    openai_api_key = "EMPTY", 
    openai_api_base = f"http://localhost:{port}/v1"
    # Set OpenAI's API key and API base to use vLLM's API server.
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base,)
    def get_judgement(item, meta_info, fw, model_name=''):
        chat_response = client.Completion.create(
            max_tokens=args.max_new_tokens,
            top_p=args.top_p,
            use_beam_search=False,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            include_stop_str_in_output = True,
            model=model_name,
            prompt=item,
            n=1, 
        )
        try:
            content = chat_response.dict()['choices'][0]['text']
            result = {
                'prompt': item,
                'gen': content,
            }
            for k in ['test_id', 'chosen', 'tag']:
                if k in meta_info.keys():
                    result[k] = meta_info[k]
            fw.write(json.dumps(result)+'\n')
            fw.flush()
        except:
            result = {}
            print('error')
        return result
    
    def update_progress_bar(done, total):
        # Simple text-based progress bar
        progress = int(50 * done / total)  # Calculate progress (50 chars width)
        sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
        sys.stdout.flush()
    
    os.makedirs(args.output_path, exist_ok=True)
    num_threads = 8
    num_files = 8
    fw_list = [open(os.path.join(args.output_path, f'{port}.{i}.jsonl'), 'w') for i in range(num_threads)]
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks and hold their futures in a list
        future_to_index = {executor.submit(get_judgement, item, meta_info[i], fw_list[i % num_files], model_name): i for i, item in enumerate(data)}
        results = {}
        # As tasks complete, update progress and store results in the original order
        done_tasks = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
            done_tasks += 1
            update_progress_bar(done_tasks, len(data))
    return results

def batch_generate_vllm(args):
    # configure strategy
    class Empty:
        pass
    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args

    # configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)
    
    prompts_data = load_data(args.dataset, is_test=False, is_arrow = True, max_samples=args.max_samples)
    if args.dataset2 is not None:
        data2 = load_data(args.dataset2, is_test=False, is_arrow = True, max_samples=args.max_samples)
        prompts_data = concatenate_datasets([prompts_data, data2])
        args.dataset2 = len(data2)

    if args.iter is None:
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    else:
        # for iterative generation
        start_idx = args.iter * args.rollout_batch_size
        end_idx = start_idx + args.rollout_batch_size
        prompts_data = prompts_data.select(range(start_idx, min(end_idx, len(prompts_data))))

    print('preprocess dataset..........')
    prompts_dataset = MyPromptDataset(prompts_data, tokenizer, dummy_strategy, input_template=args.input_template,)
    prompts = prompts_dataset.prompts
    meta_info = prompts_dataset.meta_info

    # Conditional SFT inference
    if args.enable_ca:
        for i in range(len(prompts)):
            prompts[i] += args.ca_prompt.strip() + " "

    # best of n
    N = args.best_of_n
    output_dataset = []

    print('start generation..........')
    for _ in range(N):
        process1 = multiprocessing.Process(target=predict, args=(prompts[:len(prompts)//2], 8049, args, 'model_1', meta_info[:len(prompts)//2]))
        process2 = multiprocessing.Process(target=predict, args=(prompts[len(prompts)//2:], 8050, args, 'model_2', meta_info[len(prompts)//2:]))
        process1.start()
        process2.start()
        process1.join()
        process2.join()
    
    num_threads = 8
    writer = jsonlines.open(args.output_path + '.jsonl', mode="w")
    for port in [8049,8050]:
        f_list = [open(os.path.join(args.output_path + '.jsonl', f'{port}.{i}.jsonl')) for i in range(num_threads)]
        for i in range(len(num_threads)):
            with f_list[i] as reader, jsonlines.open(args.output_path + '.jsonl', mode="a") as writer:
                for obj in reader:
                    writer.write_all(obj)
    
# def batch_generate_vllm1(args):
#     from vllm import LLM, SamplingParams

#     # configure strategy
#     class Empty:
#         pass
#     os.environ['MASTER_PORT'] = '12355' 
#     dummy_strategy = Empty()
#     dummy_strategy.print = print
#     dummy_strategy.is_rank_0 = lambda: True
#     dummy_strategy.args = args

#     # configure tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)

#     # configure model
#     print('loading llm.................')
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#     llm = LLM(model=args.pretrain, tensor_parallel_size=args.tp_size, trust_remote_code=True, seed=args.seed, )
#     os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
#     llm2 = LLM(model=args.pretrain, tensor_parallel_size=args.tp_size, trust_remote_code=True, seed=args.seed, )

#     print('llm initialized.................')
#     # Create a sampling params object.
#     sampling_params = SamplingParams(
#         max_tokens=args.max_new_tokens,
#         top_p=args.top_p,
#         use_beam_search=False,
#         temperature=args.temperature,
#         repetition_penalty=args.repetition_penalty,
#         include_stop_str_in_output = True
#     )

#     prompts_data = load_data(args.dataset, is_test=False, is_arrow = True, max_samples=args.max_samples)
#     if args.dataset2 is not None:
#         data2 = load_data(args.dataset2, is_test=False, is_arrow = True, max_samples=args.max_samples)
#         prompts_data = concatenate_datasets([prompts_data, data2])
#         args.dataset2 = len(data2)
#     # prompts_data = blending_datasets(
#     #     args.dataset,
#     #     args.dataset_probs,
#     #     dummy_strategy,
#     #     args.seed,
#     #     return_eval=False,
#     #     max_count=args.max_samples,
#     # )
#     if args.iter is None:
#         prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
#     else:
#         # for iterative generation
#         start_idx = args.iter * args.rollout_batch_size
#         end_idx = start_idx + args.rollout_batch_size
#         prompts_data = prompts_data.select(range(start_idx, min(end_idx, len(prompts_data))))

#     print('preprocess dataset..........')
#     prompts_dataset = MyPromptDataset(prompts_data, tokenizer, dummy_strategy, input_template=args.input_template)
#     prompts = prompts_dataset.prompts
#     meta_info = prompts_dataset.meta_info

#     # Conditional SFT inference
#     if args.enable_ca:
#         for i in range(len(prompts)):
#             prompts[i] += args.ca_prompt.strip() + " "

#     # best of n
#     N = args.best_of_n
#     output_dataset = []

#     def process(prompts, llm):
#         outputs = llm.generate(prompts, sampling_params)
#         for idx, output in enumerate(outputs):
#             prompt = output.prompt
#             output = output.outputs[0].text
#             result = {"prompt": prompt, "gen": output}
#             for k in ['tag', 'test_id', 'chosen']:
#                 if k in result.keys():
#                     result[k] = meta_info[idx][k]
#             output_dataset.append(result)
#         return output_dataset

#     print('start generation..........')
#     for _ in range(N):
#         output_dataset = process(prompts, llm) 

#     with jsonlines.open(args.output_path, mode="w") as writer:
#         writer.write_all(output_dataset)

def batch_generate(args):
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

    # prepare models
    model = strategy.prepare(model)
    model.eval()

    # tokenizer
    def tokenize_fn(texts):
        batch = tokenizer(
            texts,
            return_tensors="pt",
            max_length=args.prompt_max_len,
            padding=True,
            truncation=True,
        )
        return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

    # prompts_data = blending_datasets(
    #     args.dataset,
    #     args.dataset_probs,
    #     strategy,
    #     args.seed,
    #     return_eval=False,
    #     max_count=args.max_samples,
    # )
    # print('loading_dataset')
    prompts_data = load_data(args.dataset, is_test=False, is_arrow = True)
    if args.iter is None:
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    else:
        # for iterative generation
        start_idx = args.iter * args.rollout_batch_size
        end_idx = start_idx + args.rollout_batch_size
        prompts_data = prompts_data.select(range(start_idx, min(end_idx, len(prompts_data))))

    prompts_dataset = MyPromptDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset, args.micro_batch_size, True, False, drop_last=False
    )
    pbar = tqdm(
        prompts_dataloader,
        disable=not strategy.is_rank_0(),
    )

    dist.barrier()
    N = args.best_of_n
    output_dataset = []

    for prompts in pbar:
        # Conditional SFT inference
        if args.enable_ca:
            for i in range(len(prompts)):
                prompts[i] += args.ca_prompt.strip() + " "

        inputs = tokenize_fn(prompts)
        for _ in range(N):
            outputs = model.model.generate(
                **inputs,
                use_cache=True,
                max_length=args.max_len,
                do_sample=not args.greedy_sampling,
                top_p=args.top_p,
                early_stopping=True,
                num_beams=1,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for prompt, output in zip(prompts, outputs):
                output = output[len(prompt) :]
                output_dataset.append({"input": prompt, "output": output})

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
            os.remove(file)
        if args.dataset2 is not None:
            output_dataset, output_dataset2 = output_dataset[:-args.dataset2], output_dataset[-args.dataset2:]
        jsonlines.open(args.output_path, mode="w").write_all(output_dataset)
        if args.dataset2 is not None:
            jsonlines.open(args.output_path+'.2', mode="w").write_all(output_dataset2)


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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--best_of_n", type=int, default=1)
    parser.add_argument("--input_template", type=str, default="Human:\n{}\nAssistant:\n")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--length_penalty", type=float, default=1.5)
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

    args = parser.parse_args()
    if args.eval_task and args.eval_task == "generate":
        batch_generate(args)
    elif args.eval_task and args.eval_task == "generate_vllm":
        batch_generate_vllm(args)
    elif args.eval_task and args.eval_task == "rm":
        batch_rm_inference(args)
    else:
        print("Invalid or missing '--eval_task' argument. Please specify either 'generate' or 'rm'.")
