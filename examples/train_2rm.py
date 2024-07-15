import argparse
import math
import os
from datetime import datetime
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#print()
from transformers.trainer import get_scheduler
from openrlhf.datasets.utils import read_all_shard_and_evaluate
from openrlhf.datasets import PairwiseRewardDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.trainer import RewardModelTrainer2
from openrlhf.utils import prepare_dataset, get_strategy, get_tokenizer
import gc
from deepspeed.accelerator import get_accelerator
gc.collect()
get_accelerator().empty_cache()
import torch.distributed as dist

def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    strategy.print("starting..")
    # configure model
    # load huggingface model/config
    model, model2, critic_model = None, None, None
    if 's1' in args.mode:
        model = get_llm_for_sequence_regression( # s1 model
            args.pretrain,
            "reward",
            use_flash_attention_2=args.flash_attn,
            bf16=args.quantized_type,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            init_value_head=True,
        )
    if 's2' in args.mode:
        model2 = get_llm_for_sequence_regression( # s2 model
            args.pretrain,
            "reward",
            use_flash_attention_2=args.flash_attn,
            bf16=args.quantized_type,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            init_value_head=True,
        )
    if 's2' in args.mode and 'c' in args.mode:
        critic_model = Actor( # critic_model
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.quantized_type=='bf16',
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            ds_config=strategy.get_ds_train_config(is_actor=True),
        )

    strategy.print("loaded..")
    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model if 's1' in args.mode else model2, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    strategy.print(model)

    strategy.print("preprare datasets ...")
    # prepare for data and dataset
    train_data, eval_data, test_data = prepare_dataset(args.dataset, strategy)

    train_data = train_data.select(range(min(args.max_samples, len(train_data))))

    # ziyi
    # train_data, eval_data, test_data = train_data.select(range(10)), eval_data.select(range(10)), test_data.select(range(10))

    train_dataset = PairwiseRewardDataset(train_data, tokenizer, args.max_len, strategy)
    eval_dataset = PairwiseRewardDataset(eval_data, tokenizer, args.max_len, strategy)
    test_dataset = PairwiseRewardDataset(test_data, tokenizer, args.max_len, strategy)

    train_dataloader = strategy.setup_dataloader(train_dataset, args.micro_train_batch_size, True, True, train_dataset.collate_fn,)
    eval_dataloader = strategy.setup_dataloader(eval_dataset, args.micro_train_batch_size, True, False, eval_dataset.collate_fn)

    # scheduler
    num_update_steps_per_epoch = len(train_dataloader) * args.max_epochs // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        if model is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant})
        if model2 is not None:
            model2.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant})

    def initialize_optim_scheduler(model, model2=None):
         # configure optimizer
        optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)
        scheduler = get_scheduler(
            "linear",
            optim,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
        )
        if model2 is None:
            return strategy.prepare((model, optim, scheduler))
        optim2 = strategy.create_optimizer(model2, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)
        scheduler2 = get_scheduler(
            "linear",
            optim2,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
        )
        return strategy.prepare(
            (model, optim, scheduler),
            (model2, optim2, scheduler2),
        )

    optim_c, scheduler_c = None, None
    if model is not None:
        (model, optim, scheduler) = initialize_optim_scheduler(model)
    if critic_model is not None and model2 is not None:
        (critic_model, optim_c, scheduler_c), (model2, optim, scheduler) = initialize_optim_scheduler(critic_model, model2)
    elif model2 is not None:
        (model2, optim, scheduler) = initialize_optim_scheduler(model2)
    
    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is micro_batch_size * 2
    # we use merged chosen + rejected response forward
    trainer = RewardModelTrainer2(
        model=model,
        model2=model2,
        critic_model=critic_model,
        strategy=strategy,
        optim=optim,
        optim_c = optim_c,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        scheduler_c=scheduler_c,
        max_norm=args.max_norm,
        max_epochs=args.max_epochs,
        loss=args.loss,
        mode = args.mode
    )

    trainer.fit(args)

    test_dataloader = strategy.setup_dataloader(test_dataset, args.micro_train_batch_size, True, False, eval_dataset.collate_fn)
    local_rank = dist.get_rank()
    os.makedirs(os.path.join(args.save_path, 'results/'), exist_ok=True)
    output_fname = os.path.join(os.path.join(args.save_path, 'results/'), f"predict_{local_rank}.json")
    output_file = open(output_fname, "w")
    trainer.test(test_dataloader, get_token_info=False, output_file=output_file)

    if local_rank == 0:
        read_all_shard_and_evaluate(os.path.join(args.save_path, 'results/'))
    # save model checkpoint after fitting on only rank0
    trainer.model_list.save(strategy, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="bigscience/bloomz-1b7")
    # parser.add_argument('--dataset', type=str, default='Anthropic/hh-rlhf')
    parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_rm")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--loss", type=str, default="sigmoid")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--quantized_type", type=str, default='bf16')
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--compute_fp32_loss", action="store_true", default=False)
    parser.add_argument("--margin_loss", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=list, default=None)
    parser.add_argument("--input_template", type=str, default="Human: {}\nAssistant: ")
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    parser.add_argument("--bos_token", type=str, default=None)
    parser.add_argument("--eos_token", type=str, default=None)
    parser.add_argument("--pad_token", type=str, default=None)
    parser.add_argument("--unk_token", type=str, default=None)

    # custom dataset key name
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default=None)
    parser.add_argument("--rejected_key", type=str, default=None)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_rm")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="rm_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    parser.add_argument("--mode", type=str, default="s1")

    torch.cuda.empty_cache()
    args = parser.parse_args()
    train(args)
