import logging
import os
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import jsonlines
import sys
import re
import glob
import shutil
from typing import Dict, List, Tuple
from transformers import (
    PreTrainedTokenizer,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


class CMLDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str,
                 block_size):

        assert os.path.isfile(file_path)

        logger.info("Creating features from dataset file at %s", file_path)

        self.examples = []
        with jsonlines.open(file_path, 'r') as f:
            for ix, line in tqdm(enumerate(f), desc="Reading Jsonlines", ascii=True):
                if (None in line["inputids"]) or (
                        None in line["labels"]):
                    print("LineNum:", ix)
                    continue
                else:
                    self.examples.append(line)

        self.block_size = int(block_size)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        block_size = self.block_size
        tokenizer = self.tokenizer

        item = self.examples[i]

        input_ids = item["inputids"]
        labels = item["labels"]
        subword_labels = item["subword_labels"]
        fids = item["fid"]
        dataorigin_labels = item["dataorigin_labels"]


        assert len(input_ids) == len(labels)
        assert len(input_ids) == len(subword_labels)
        assert len(input_ids) == len(dataorigin_labels)

        if len(input_ids) >= block_size - 2:
            input_ids = input_ids[0:block_size - 2]
            labels = labels[0:block_size - 2]
            subword_labels = subword_labels[0:block_size - 2]
            dataorigin_labels = dataorigin_labels[0:block_size - 2]

        elif len(input_ids) < block_size - 2:
            input_ids = input_ids + [tokenizer.pad_token_id] * (self.block_size - 2 - len(input_ids))
            labels = labels + [tokenizer.pad_token_id] * (self.block_size - 2 - len(labels))
            subword_labels = subword_labels + [tokenizer.pad_token_id] * (self.block_size - 2 - len(subword_labels))
            dataorigin_labels = dataorigin_labels + [tokenizer.pad_token_id] * (self.block_size - 2 - len(dataorigin_labels))

        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        labels = tokenizer.build_inputs_with_special_tokens(labels)


        subword_labels = [-100] + subword_labels + [-100]
        dataorigin_labels = [-100] + dataorigin_labels + [-100]

        assert len(input_ids) == len(labels)
        assert len(input_ids) == block_size
        assert len(input_ids) == len(subword_labels)
        assert len(input_ids) == len(dataorigin_labels)
        try:
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            subword_labels = torch.tensor(subword_labels, dtype=torch.long)
            dataorigin_labels = torch.tensor(dataorigin_labels, dtype=torch.long)

            mask_idxs = (input_ids == tokenizer.mask_token_id).bool()
            labels[~mask_idxs] = -100
            labels = labels.reshape(labels.size()[0], 1)
            subword_labels = subword_labels.reshape(subword_labels.size()[0], 1)
            dataorigin_labels = dataorigin_labels.reshape(dataorigin_labels.size()[0], 1)

            labels = torch.cat((labels, subword_labels,dataorigin_labels), -1)


        except:
            print("Idx:", i, "Len of Examples:", len(self.subwords))
            print("Unexpected error:", sys.exc_info()[0])

            raise

        return input_ids, labels,fids


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))


    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted



def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    print("checkpoints_sortedcheckpoints_sorted:", checkpoints_sorted)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)



