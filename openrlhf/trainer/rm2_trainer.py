import math
from abc import ABC

import loralib as lora
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
from openrlhf.models import LogExpLoss, PairWiseLoss, SwitchBalancingLoss
import json
import os
import numpy as np
import copy

class Model_list:
    def __init__(self, model_list):
        self.model_list = model_list
        self.model_names = ['s1','s2','c']
    def train(self,):
        for model in self.model_list:
            if model is not None:
                model.train()
    def eval(self,):
        for model in self.model_list:
            if model is not None:
                model.eval()
    def step(self, strategy, loss, optimizer, scheduler):
        for model in self.model_list:
            if model is not None:
                strategy.backward(loss, model, optimizer)
                strategy.optimizer_step(optimizer, model, scheduler)
    def save(self, strategy, tokenizer, path):
        for idx, model in enumerate(self.model_list):
            if model is not None:
                os.makedirs(path + f'/{self.model_names[idx]}', exist_ok=True)
                strategy.save_model(model, tokenizer, path + f'/{self.model_names[idx]}')

class RewardModelTrainer2(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        model2,
        critic_model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 2,
        loss="sigmoid",
        mode = 's1'
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.model2 = model2
        self.critic_model = critic_model
        self.model_list = Model_list([self.model, self.model2, self.critic_model])
        self.mode = mode
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.loss_fn = nn.MSELoss()

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(range(self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(self.epochs):
            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model_list.train()
            acc_mean1 = 0
            acc_mean2 = 0
            loss_mean = 0

            for prompt_ids, prompt_mask, gen_ids, gen_mask, chosens, extra, meta_info, prompts_id_len in self.train_dataloader:
                prompt_ids = prompt_ids.squeeze(1).to(torch.cuda.current_device())
                prompt_mask = prompt_mask.squeeze(1).to(torch.cuda.current_device())
                gen_ids = gen_ids.squeeze(1).to(torch.cuda.current_device())
                gen_mask = gen_mask.squeeze(1).to(torch.cuda.current_device())
                chosens = chosens.to(torch.cuda.current_device())

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None
                loss, s1_loss, s2_loss, c_loss, all_values1, all_values2 = 0, 0, 0, 0, None, None
                if 's1' in args.mode:
                    all_values1, output = self.model(prompt_ids, attention_mask=prompt_mask, return_output=True)
                    if self.compute_fp32_loss:
                        all_values1 = all_values1.float()
                    s1_loss = self.loss_fn(all_values1, chosens)
                    loss += s1_loss
                    s1_loss = s1_loss.item()
                if 's2' in args.mode and 'c' not in args.mode:
                    all_values2, output = self.model2(gen_ids, attention_mask=gen_mask, return_output=True)
                    if self.compute_fp32_loss:
                        all_values2 = all_values2.float()
                    s2_loss = self.loss_fn(all_values2, chosens)
                    loss += s2_loss
                    s2_loss = s2_loss.item()
                if 'c' in args.mode:
                    labels = torch.where(
                        gen_mask.bool(),
                        gen_ids,
                        self.loss_fn.IGNORE_INDEX,
                    )
                    for label, source_len in zip(labels, prompts_id_len):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX
                    output = self.model(inputs, attention_mask=attention_mask, return_output=True)
                    loss += c_loss
                    c_loss = c_loss.item()
                if 's2' in args.mode and 'c' in args.mode:
                    pass


                if not self.aux_loss:
                    aux_loss = 0
                loss += aux_loss * self.args.aux_loss_coef
                self.model_list.step(self.strategy, loss, self.optimizer, self.scheduler)
                if all_values1 is not None:
                    acc_mean1 = acc_mean1 * 0.9 + 0.1 * ((torch.abs(all_values1 - chosens) < 0.5).int()).float().mean().item()
                if all_values2 is not None:
                    acc_mean2 = acc_mean2 * 0.9 + 0.1 * ((torch.abs(all_values2 - chosens) < 0.5).int()).float().mean().item()
                # TODO: Why print this as loss? Whay smooth all the loss?
                loss_mean = loss_mean * 0.9 + 0.1 * loss.item()
                # optional rm info
                logs_dict = {
                    "loss": loss.item(),
                    's1_loss': s1_loss,
                    's2_loss': s2_loss,
                    "acc_mean1": acc_mean1,
                    "acc_mean2": acc_mean2,
                    "loss_mean": loss_mean,
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                # logs/checkpoints/evaluate
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.model_list.save(self.strategy, self.tokenizer, os.path.join(args.ckpt_path, tag))
            # self.strategy.save_ckpt(self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)

    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        self.model_list.eval()
        with torch.no_grad():
            acc1 = 0
            acc2 = 0
            loss_sum = 0
            s1_loss_sum = 0
            s2_loss_sum = 0

            for prompt_ids, prompt_mask, gen_ids, gen_mask, chosens, extra, meta_info, prompt_ids_len in eval_dataloader:
                prompt_ids = prompt_ids.squeeze(1).to(torch.cuda.current_device())
                prompt_mask = prompt_mask.squeeze(1).to(torch.cuda.current_device())
                gen_ids = gen_ids.squeeze(1).to(torch.cuda.current_device())
                gen_mask = gen_mask.squeeze(1).to(torch.cuda.current_device())
                chosens = chosens.to(torch.cuda.current_device())

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None
                loss, s1_loss, s2_loss, all_values1, all_values2 = 0, 0, 0, None, None
                if 's1' in self.args.mode:
                    all_values1, output = self.model(prompt_ids, attention_mask=prompt_mask, return_output=True)
                    if self.compute_fp32_loss:
                        all_values1 = all_values1.float()
                    s1_loss = self.loss_fn(all_values1, chosens)
                    loss += s1_loss
                    s1_loss_sum += s1_loss.item()
                if 's2' in self.args.mode:
                    all_values2, output = self.model2(gen_ids, attention_mask=gen_mask, return_output=True)
                    if self.compute_fp32_loss:
                        all_values2 = all_values2.float()
                    s2_loss = self.loss_fn(all_values2, chosens)
                    loss += s2_loss
                    s2_loss_sum += s2_loss.item()

                acc1 += ((torch.abs(all_values1 - chosens) < 0.5).int()).float().mean().item() if all_values1 is not None else 0
                acc2 += ((torch.abs(all_values2 - chosens) < 0.5).int()).float().mean().item() if all_values2 is not None else 0

                loss_sum += loss.item()
                step_bar.update()

            acc_mean1 = acc1 / self.eval_dataloader.__len__()
            acc_mean2 = acc2 / self.eval_dataloader.__len__()
            loss_mean = loss_sum / self.eval_dataloader.__len__()
            
            if all_values1 is not None:
                all_values = self.strategy.all_gather(all_values1)
                reward_mean = torch.mean(all_values)
                reward_std = torch.std(all_values).clamp(min=1e-8)
                self.strategy.print("Set reward mean std")
                unwrap_model = self.strategy._unwrap_model(self.model)
                unwrap_model.config.mean = reward_mean.item()
                unwrap_model.config.std = reward_std.item()
            if all_values2 is not None:
                all_values = self.strategy.all_gather(all_values2)
                reward_mean = torch.mean(all_values)
                reward_std = torch.std(all_values).clamp(min=1e-8)
                self.strategy.print("Set reward mean std")
                unwrap_model = self.strategy._unwrap_model(self.model2)
                unwrap_model.config.mean = reward_mean.item()
                unwrap_model.config.std = reward_std.item()

            bar_dict = {
                "eval_loss": loss_mean,
                "acc_mean1": acc_mean1,
                "acc_mean2": acc_mean2,
                's1_loss': s1_loss_sum,
                's2_loss': s2_loss_sum,
            }
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
        self.model_list.train()  # reset model state

    def test(self, eval_dataloader, get_token_info = False, output_file=None):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Test stage...",
            disable=not self.strategy.is_rank_0(),
        )
        all_info_dict_mode = {'loss':[], 'acc1':[], 'acc2':[],'tags':[], 'test_id': [], 's1_loss':[], 's2_loss':[]}
        self.model_list.eval()
        with torch.no_grad():
            acc1 = 0
            acc_list1 = []
            acc2 = 0
            acc_list2 = []
            tags = []
            loss_sum = 0
            for prompt_ids, prompt_mask, gen_ids, gen_mask, chosens, extra, meta_infos, prompt_ids_len in eval_dataloader:
                prompt_ids = prompt_ids.squeeze(1).to(torch.cuda.current_device())
                prompt_mask = prompt_mask.squeeze(1).to(torch.cuda.current_device())
                gen_ids = gen_ids.squeeze(1).to(torch.cuda.current_device())
                gen_mask = gen_mask.squeeze(1).to(torch.cuda.current_device())
                chosens = chosens.to(torch.cuda.current_device())

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None
                s1_loss_sum, s2_loss_sum = 0, 0
                loss, s1_loss, s2_loss, all_values1, all_values2 = 0, 0, 0, None, None
                if 's1' in self.args.mode:
                    all_values1, output = self.model(prompt_ids, attention_mask=prompt_mask, return_output=True)
                    if self.compute_fp32_loss:
                        all_values1 = all_values1.float()
                    s1_loss = self.loss_fn(all_values1, chosens)
                    loss += s1_loss
                    s1_loss_sum += s1_loss.item()
                if 's2' in self.args.mode:
                    all_values2, output = self.model2(gen_ids, attention_mask=gen_mask, return_output=True)
                    if self.compute_fp32_loss:
                        all_values2 = all_values2.float()
                    s2_loss = self.loss_fn(all_values2, chosens)
                    loss += s2_loss
                    s2_loss_sum += s2_loss.item()
                all_info_dict = copy.deepcopy(all_info_dict_mode)
                if all_values1 is not None:
                    acc_bool1 = (torch.abs(all_values1 - chosens) < 0.5).int()
                    acc1 += acc_bool1.float().mean().item() 
                    acc_list1.extend(acc_bool1.float().flatten().tolist())
                    all_info_dict['acc1'] = (acc_bool1.float().flatten().tolist())
                if all_values2 is not None:
                    acc_bool2 = (torch.abs(all_values2 - chosens) < 0.5).int()
                    acc2 += acc_bool2.float().mean().item()
                    acc_list2.extend(acc_bool2.float().flatten().tolist())
                    all_info_dict['acc2'] = (acc_bool2.float().flatten().tolist())

                tags.extend([meta["tag"] for meta in meta_infos])

                loss_sum += loss.mean().item()
                step_bar.update()

                
                all_info_dict['tag'] = ([meta["tag"] for meta in meta_infos])
                all_info_dict['test_id'] = ([meta["test_id"] for meta in meta_infos])
                all_info_dict['loss'] = (loss.detach().cpu().tolist()) 
                all_info_dict['s1_loss'] = s1_loss_sum
                all_info_dict['s2_loss'] = s2_loss_sum

                output_file.write(json.dumps(all_info_dict,) + '\n')
                output_file.flush()
               

            self.strategy.print(f"Test Data Size: {eval_dataloader.__len__()}")
            acc_mean1 = acc1 / eval_dataloader.__len__()
            acc_mean2 = acc2 / eval_dataloader.__len__()
            loss_mean = loss_sum / eval_dataloader.__len__()

            bar_dict = {
                f"eval_loss": loss_mean,
                f"acc_mean1": acc_mean1,
                f"acc_mean2": acc_mean2
            }
            self.strategy.print(bar_dict)
            from collections import defaultdict
            for idx, acc in enumerate([acc1, acc2]):
                self.strategy.print(f"===========Detail ACC {idx}==========")
                acc_dict = defaultdict(list)
                overall_acc = []
                for tag, acc in zip(tags, acc_list1):
                    acc_dict[tag].append(acc)
                    overall_acc.append(acc)

                sort_keys = sorted(list(acc_dict.keys()))
                for tag in sort_keys:
                    accs = acc_dict[tag]
                    self.strategy.print(f"{tag}:\t\t{np.mean(accs):.4f}\t\tdata length:{len(accs)}")

                self.strategy.print(f"Overall:\t\t{np.mean(overall_acc):.4f}")

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, return_all = False):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)

        all_values, output = model(input_ids, attention_mask=att_masks, return_output=True)
        chosen_rewards = all_values[: chosen_ids.shape[0]]
        rejected_rewards = all_values[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        if return_all:
            return chosen_rewards, rejected_rewards, aux_loss, output
        else:
            return chosen_rewards, rejected_rewards, aux_loss

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks
