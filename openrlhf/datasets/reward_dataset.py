from typing import Callable
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none, zero_pad_sequences

def process_reward_data(tokenizer, data):
    messages = []
    for i, message in enumerate(data['conversations']):
        if i % 2 == 0:
            assert message["from"] == "human"
            messages.append({"role": "user", "content": message["value"]})
        else:
            assert message["from"] == "gpt"
            messages.append({"role": "assistant", "content": message["value"]})
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

    chosen, rejected, count, tag, data_type = data["chosen"], data["rejected"], data["count"], data["tag"], data["type"]
    meta_info = {"count": count, "tag": tag, "data_type":data_type}

    chosen = tokenizer.apply_chat_template([{"role": "assistant", "content":chosen}], add_generation_prompt=False, tokenize=False)
    rejected = tokenizer.apply_chat_template([{"role": "assistant", "content":rejected}], add_generation_prompt=False, tokenize=False)

    return prompt, chosen, rejected, meta_info

def preprocess_data(data, input_template=None, prompt_key=None, chosen_key=None, rejected_key=None) -> str:
    system_prompt = None

    # custom dataset
    if chosen_key and rejected_key:
        if prompt_key:
            prompt = data[prompt_key]
        else:
            prompt = ""
            input_template = None  # do not modified with input template again
        chosen = data[chosen_key]
        reject = data[rejected_key]
    else:
        # Anthropic/hh-rlhf
        # tasksource/oasst1_pairwise_rlhf_reward
        if exist_and_not_none(data, "chosen") and exist_and_not_none(data, "rejected"):
            prompt = data["prompt"] if exist_and_not_none(data, "prompt") else ""
            if prompt.startswith("prompter:"):
                prompt = (
                    prompt.replace("prompter:", "\nHuman:\n").replace("assistant:", "\nAssistant:\n")
                    + "\nAssistant:\n"
                )
            chosen = data["chosen"]
            reject = data["rejected"]
            input_template = None  # do not modified with input template again
        # lmsys/chatbot_arena_conversations
        elif exist_and_not_none(data, "winner") and exist_and_not_none(data, "conversation_a"):

            def process_chatbot_arena_conversations(lll):
                result = []
                for l in lll:
                    if "user" in l["role"]:
                        result.append(input_template.format(l["content"]))
                    else:
                        result.append(l["content"] + "\n")
                return "".join(result)

            prompt = ""
            chosen = data["conversation_a"] if data["winner"] == "model_a" else data["conversation_b"]
            reject = data["conversation_b"] if data["winner"] == "model_a" else data["conversation_a"]
            chosen = process_chatbot_arena_conversations(chosen)
            reject = process_chatbot_arena_conversations(reject)
            input_template = None  # do not modified with input template again
        # openai/webgpt_comparisons
        elif exist_and_not_none(data, "answer_0") and exist_and_not_none(data, "answer_1"):
            prompt = data["question"]["full_text"]
            chosen = data["answer_0"] if data["score_0"] > data["score_1"] else data["answer_1"]
            reject = data["answer_1"] if data["score_0"] > data["score_1"] else data["answer_0"]
        else:
            raise ValueError("Unknown reward dataset")

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    # input template
    if input_template:
        prompt = input_template.format(prompt)

    if system_prompt:
        prompt = system_prompt + "\n" + prompt
    return prompt, chosen, reject, margin


class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        is_dpo=False,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo

        self.prompts = []
        self.chosens = []
        self.rejects = []
        self.meta_infos = []

        if self.is_dpo:
            self.prompt_ids_lens = []
        else:
            self.margins = []

        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):                
            prompt, chosen, reject, meta_info = process_reward_data(self.tokenizer, data)
            margin = 0
            #prompt, chosen, reject, margin = preprocess_data(
            #    data, input_template, prompt_key, chosen_key, rejected_key
            #)

            # prompt_ids_len for prompt mask
            if self.is_dpo:
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
                # filter the sample whose length is greater than max_length (2 for answer length)
                if prompt_ids_len >= self.max_length - 2:
                    continue
                else:
                    self.prompt_ids_lens.append(prompt_ids_len)
            else:
                self.margins.append(margin)

            self.prompts.append(prompt)
            self.chosens.append(chosen)
            self.rejects.append(reject)
            self.meta_infos.append(meta_info)

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject, meta_info = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.meta_infos[idx]
        if self.is_dpo:
            extra = self.prompt_ids_lens[idx]
        else:
            extra = self.margins[idx]

        chosen = prompt + chosen
        chosen = chosen.rstrip()

        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        reject = prompt + reject
        reject = reject.rstrip()

        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            extra,
            meta_info
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        extras = []
        meta_infos = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra, meta_info in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            extras.append(extra)
            meta_infos.append(meta_info)

        chosen_ids = zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks)
        reject_ids = zero_pad_sequences(reject_ids, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, extras, meta_infos


class PairwiseRewardDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        is_dpo=False,
        args=None,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.chosens = []
        self.prompts, self.old_prompts = [], []
        self.gen = []
        self.meta_infos = []
        self.args=args
        if self.is_dpo:
            self.prompt_ids_lens = []
        else:
            self.margins = []

        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):   
            gen = data['gen']       
            meta_info = {
                'test_id': data['test_id'],
                'tag': data['tag'],
                'chosen': data['chosen']
            }
            if self.is_dpo:
                tmp_prompt = data['prompt_old'] if 'new' in self.args.mode else data['prompt']
                prompt_token = self.tokenizer(
                    tmp_prompt,
                    max_length=max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
                # filter the sample whose length is greater than max_length (2 for answer length)
                if prompt_ids_len >= self.max_length - 2:
                    continue
                self.prompt_ids_lens.append(prompt_ids_len)
            else:
                self.margins.append(0)
            if 'new' in self.args.mode:
                self.old_prompts.append(data['prompt_old'])
            self.chosens.append(int(data['chosen'] - 1))
            self.prompts.append(data['prompt'])
            self.gen.append(gen)
            self.meta_infos.append(meta_info)

    def __len__(self):
        length = len(self.meta_infos)
        return length

    def __getitem__(self, idx):
        prompt, gen, meta_info = self.prompts[idx], self.gen[idx], self.meta_infos[idx]
        if self.is_dpo:
            extra = self.prompt_ids_lens[idx]
        else:
            extra = self.margins[idx]
        prompt = prompt.rstrip()
        prompt_token = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        if 'new' in self.args.mode:
            old_prompt = self.old_prompts[idx].rstrip()
            old_prompt_token = self.tokenizer(
                old_prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            prompt_ids_len = old_prompt_token["attention_mask"].int().sum().item()
        else:
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
        
        if self.args.mode == 'c':
            prompt = (prompt + gen).rstrip()
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )

        # to avoid EOS_token truncation
        prompt_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        prompt_token["attention_mask"][0][-1] = True

        return (
            prompt_token["input_ids"],
            prompt_token["attention_mask"],
            self.chosens[idx],
            extra,
            meta_info,
            prompt_ids_len
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        extras = []
        meta_infos = []
        prompt_ids_lens = []
        chosens = []
        for chosen_id, chosen_mask, chosen, extra, meta_info, prompt_ids_len in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            extras.append(extra)
            meta_infos.append(meta_info)
            chosens.append(chosen)
            prompt_ids_lens.append(prompt_ids_len - 1)

        chosen_ids = zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks)
        if 'new' in self.args.mode:
            return chosen_ids, chosen_masks, torch.tensor(chosens, dtype=torch.bfloat16), extras, meta_infos, torch.tensor(prompt_ids_lens, dtype=torch.long)
        else:
            return chosen_ids, chosen_masks, torch.tensor(chosens, dtype=torch.bfloat16), extras, meta_infos, prompt_ids_lens
