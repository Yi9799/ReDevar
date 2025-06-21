import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.activations import gelu
import jsonlines
from tqdm import tqdm
from transformers import (
    RobertaForMaskedLM,
)

import os

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


class RobertaLMHead2(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.out_vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.out_vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x




position_dict = dict()
def get_varnum(file_path):
    max_subword = 0
    max_origin=0
    global position_dict


    with jsonlines.open(file_path, 'r') as f:
        for ix, line in tqdm(enumerate(f), desc="get_var_num", ascii=True):
            if (None in line["inputids"]) or (
                    None in line["labels"]):
                print("LineNum:", ix)
                continue
            else:
                fid=line["fid"]
                varmap_position2=line["varmap_position2"]
                position_dict[fid]=varmap_position2

                subword_labels=line["subword_labels"]
                dataorigin_labels = line["dataorigin_labels"]
                num2=max(subword_labels)
                num =max(dataorigin_labels)
                if num2>max_subword:
                    max_subword=num2
                if num>max_origin:
                    max_origin=num
    return max_subword,max_origin


class myRobertaLMHead2(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.out_class, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.out_class))

        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

class myRobertaLMHead3(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


        self.decoder = nn.Linear(config.hidden_size, config.out_class2, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.out_class2))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


def pre(input_ids,embeddings,fids):
    global position_dict
    varmap_position2s=[]
    for fid in fids:
        varmap_position2s .append(position_dict[fid])

    input_mask= input_ids == 4

    for batch_index in range(input_ids.size(0)):

        current_embeddings = embeddings[batch_index]
        varmap_position2 = varmap_position2s[batch_index]
        current_mask = input_mask[batch_index]
        identifier_indices = torch.nonzero(current_mask, as_tuple=True)[0]
        for var, positions in varmap_position2.items():
            mapped_positions = [identifier_indices[pos] for pos in positions]
            var_embeddings = torch.stack([current_embeddings[pos] for pos in mapped_positions])
            pooled = torch.mean(var_embeddings, dim=0)
            for pos in mapped_positions:
                current_embeddings[pos] = pooled

        embeddings[batch_index] = current_embeddings
    return embeddings


class RobertaForMaskedLMv2(RobertaForMaskedLM):

    def __init__(self, config):
        super().__init__(config)

        num_list2 = []
        num_list = []

        for run_type in ["train", "test","validation"]:
            file_path=os.path.join(config.dir,f"preprocessed_{run_type}_src2.json")

            max_subword,max_origin=get_varnum(file_path)
            num_list2.append(max_subword)
            num_list.append(max_origin)
        config.out_class = max(num_list2)+1
        config.out_class2 = max(num_list)+1


        self.lm_head2 = RobertaLMHead2(config)
        self.lm_head_my2 = myRobertaLMHead2(config)
        self.lm_head_my3 = myRobertaLMHead3(config)
        self.out_vocab_size = config.out_vocab_size
        self.out_class=config.out_class
        self.out_class2=config.out_class2

        self.alpha= config.alpha
        self.beta=config.beta

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        type_label=None,
            flag=None,
            fids=None

    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        var_labels = labels[:, :, 0]
        subword_labels = labels[:, :, 1]
        dataorigin_labels = labels[:, :, 2]

        loss_fct = CrossEntropyLoss()
        loss_fct_subword = CrossEntropyLoss()
        loss_fct_dataorigin = CrossEntropyLoss()
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        pre_sequence_output=pre(input_ids,sequence_output,fids)

        prediction_scores = self.lm_head2(pre_sequence_output)
        prediction_scores_subword=self.lm_head_my2(sequence_output)
        prediction_scores_dataorigin = self.lm_head_my3(sequence_output)
        masked_lm_loss = None
        masked_lm_loss_subword = None
        masked_lm_loss_dataorigin=None

        if flag=="test":

            output_pred_scores = torch.topk(prediction_scores, k=5, dim=-1)
            output_pred_scores_subword = torch.topk(prediction_scores_subword, k=1, dim=-1)
            output_pred_scores_dataorigin = torch.topk(prediction_scores_dataorigin, k=1, dim=-1)

            outputs = (output_pred_scores, output_pred_scores_subword,output_pred_scores_dataorigin)
            if labels is not None:

                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.out_vocab_size), var_labels.view(-1))
                masked_lm_loss_subword = loss_fct_subword(prediction_scores_subword.view(-1, self.out_class),
                                                    subword_labels.view(-1))
                masked_lm_loss_dataorigin = loss_fct_dataorigin(prediction_scores_dataorigin.view(-1, self.out_class2),
                                                    dataorigin_labels.view(-1))
                masked_loss = masked_lm_loss +self.alpha*masked_lm_loss_dataorigin+ self.beta*masked_lm_loss_subword
                outputs = (masked_loss,) + outputs
                return outputs
        else:

            if labels is not None:

                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.out_vocab_size), var_labels.view(-1))
                masked_lm_loss_subword= loss_fct_subword(prediction_scores_subword.view(-1, self.out_class), subword_labels.view(-1))
                masked_lm_loss_dataorigin = loss_fct_dataorigin(prediction_scores_dataorigin.view(-1, self.out_class2),
                                                          dataorigin_labels.view(-1))


            masked_loss = masked_lm_loss+self.alpha*masked_lm_loss_dataorigin+ self.beta*masked_lm_loss_subword
            output = (prediction_scores,prediction_scores_subword,prediction_scores_dataorigin) + outputs[2:]#注意不要改左加数，否则会变成张量加元组，会报错
            return ((masked_loss,) + output) if masked_loss is not None else output

