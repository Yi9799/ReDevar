import argparse
import glob
import logging
import os
import random
import re
import shutil
from typing import Dict, List, Tuple
from utils import CMLDataset,_sorted_checkpoints,_rotate_checkpoints
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import json
from model import RobertaForMaskedLMv2


from transformers import (
    WEIGHTS_NAME,
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "roberta": (RobertaConfig, RobertaForMaskedLMv2, RobertaTokenizerFast),
}



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



def l1_regularization(model, lambda_l1):
    l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
    return lambda_l1 * l1_norm


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:


    tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size

    if args.evaluate_during_training:
        eval_dataset = CMLDataset(tokenizer, args.eval_data_file,
                              args.block_size)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
                    len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    no_decay = ["bias",  "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # sgd = SGD(lr=learning_rate, decay=learning_rate/nb_epoch, momentum=0.9, nesterov=True)

    # optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate, momentum=0.9)
    scheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if (
            args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)


    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        print("args.model_name_or_path", args.model_name_or_path)
        try:

            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step =int(checkpoint_suffix)

            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")


    model.zero_grad()


    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    during_logging_loss = 0.0
    epoch_num = 0
    for _ in train_iterator:
        tr_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        logging_times = 0
        for step, batch in enumerate(epoch_iterator):

            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, labels,fids= batch
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)


            model.train()

            outputs = model(inputs, labels=labels,fids=fids,flag="train")
            loss = outputs[0]
            lambda_l1 = 0.01

            l1_loss = l1_regularization(model, lambda_l1)

            # loss = loss + l1_loss/ args.per_gpu_train_batch_size

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if torch.isnan(loss):
                print(fids)
                continue


            tr_loss += loss.item()
            during_logging_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if global_step == 2:
                    logger.info("global_step = %s, this_loss = %s,  lr=%6.1e", global_step, loss.item(),
                                scheduler.get_lr()[0])

                total_logging_times = len(train_dataloader) // args.logging_steps
                if (args.logging_steps > 0 and global_step % args.logging_steps == 0):

                    logging_mean_loss = during_logging_loss / args.logging_steps


                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss1", logging_mean_loss,
                                         global_step)
                    tb_writer.add_scalar("loss2", loss.item(), global_step)
                    logger.info("global_step = %s, this_loss = %s, logging_mean_loss = %s, lr=%6.1e",
                                global_step, loss.item(), logging_mean_loss, scheduler.get_lr()[0])

                    during_logging_loss = 0

                    logging_times += 1



                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"

                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix,
                                                                              global_step))

                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (model.module if hasattr(model,
                                                             "module") else model)
                    model_to_save.save_pretrained(output_dir,
                                                  safe_serialization=False)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break


        train_average_loss = (tr_loss) / len(train_dataloader)

        train_perplexity = torch.exp(torch.tensor(train_average_loss))

        train_result = {"train_average_loss": train_average_loss, "perplexity": train_perplexity}

        for key in sorted(train_result.keys()):
            logger.info("-------------------- train average results ---------------------")
            logger.info("  %s = %s", key, str(train_result[key]))
        if (
                args.evaluate_during_training
        ):
            results = evaluate(args, model, tokenizer, eval_dataset=eval_dataset)
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value,
                                     global_step)
            logger.info("EVAL Results:" + str(results))


        epoch_num += 1

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="", eval_dataset=None) -> Dict:

    eval_output_dir = args.output_dir

    if eval_dataset is None:
        eval_dataset = CMLDataset(tokenizer, args.eval_data_file,
                                  args.block_size)
    os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.train_batch_size

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
    )


    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        inputs, labels ,fids= batch  #
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():

            outputs = model(inputs, labels=labels,fids=fids,
                            flag="validation")
            lm_loss = outputs[0]
            if torch.isnan(lm_loss):
                print(fids)
                continue
            eval_loss += lm_loss.mean().item()

            if nb_eval_steps != 0 and nb_eval_steps % (len(eval_dataloader) //2) == 0:
                logger.info("nb_eval_steps = %s, this_loss = %s,sofar_mean_loss=%s", nb_eval_steps, lm_loss.item(),
                            eval_loss / nb_eval_steps)


        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"eval_loss": eval_loss, "perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )


    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )


    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(  #
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=800,
        type=int,
        help="Optional input sequence length after tokenization."
             "The training dataset will be truncated in block of this size for training."
             "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",

    )
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."

    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",

    )
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"

    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"

    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",

    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."  
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument("--vocab_path", type=str, default="",
                        help="Path to the out cls vocab")
    parser.add_argument("--alpha", type=int, default=1, help="random seed for initialization")
    parser.add_argument("--beta", type=int, default=8, help="random seed for initialization")#16

    args = parser.parse_args()


    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]
        logger.info("Starting from Checkpoint: " + args.model_name_or_path)

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


    set_seed(args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    vocab = json.load(open(args.vocab_path))
    config.out_vocab_size = len(vocab)+1

    (config.dir, filename) = os.path.split(args.train_data_file)
    config.alpha=args.alpha
    config.beta=args.beta

    # config.hidden_dropout_prob=0.1
    # config.attention_probs_dropout_prob=0.1
    # config.layer_norm_eps=1e-6

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence

    else:
        args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,

        )
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)


    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = CMLDataset(tokenizer, args.train_data_file,
                                   args.block_size)  # load_and_cache_examples(args, tokenizer, evaluate=False)#这里面可直接加载grammar
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" final total train ---------------global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train:

        os.makedirs(args.output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = (model.module if hasattr(model, "module") else model)  # 如果model是一个多GPU的模型（即model有module属性）
        model_to_save.save_pretrained(args.output_dir, safe_serialization=False)
        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        model = model_class.from_pretrained(args.output_dir)

        tokenizer = tokenizer_class.from_pretrained(args.output_dir)

        logger.info(" model and tokenizer loaded from checkpoint to %s", args.output_dir)
        model.to(args.device)


    results = {}
    if args.do_eval:
        checkpoints = [args.output_dir]  # ?
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()