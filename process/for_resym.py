
import argparse
import json
import os
from tqdm import tqdm
import sys
import re
from collections import defaultdict

var_name_to_pred=dict()
type_stripped_vars=dict()

def replace_names(match):

    var_id = match.group(1)
    original_name=match.group(2)

    global type_stripped_vars,var_name_to_pred
    var_name_to_pred.clear()
    assert var_name_to_pred
    origin=type_stripped_vars[original_name]
    if origin=="ida" or origin=="ghidra":
        return original_name
    if origin=="dwarf":
        var_name_to_pred[var_id]=original_name
        return var_id



def rewrite(args):

    dir =  os.path.dirname(args.outpath)
    if not os.path.exists(dir):  #
        os.makedirs(dir)

    pattern = r"@@(\w+)@@(\w+)@@"
    with open(args.inpath,"r") as f, open(args.outpath,"w")  as new_f:
        for ix, line in tqdm(enumerate(f), desc="Reading Jsonlines", ascii=True):
            line = json.loads(line)


            fid = str(line['id']) + "-" + line['func_name'].replace("_", "-")

            global type_stripped_vars
            type_stripped_vars = line["type_stripped_vars"]

            func = line["func"]
            updated_func = re.sub(pattern, replace_names, func)

            new_line = dict()
            global var_name_to_pred
            to_predicts='`, `'.join(var_name_to_pred.keys())
            input = f"What are the original name of variables `{to_predicts}`?\n```\n{updated_func}\n```"
            new_line["input"]=input
            formatted_strings = []
            for var,original_name in var_name_to_pred.items():

                formatted_strings.append(f"{var}: {original_name}")
            var_name_to_pred.clear()
            output = "\n".join(formatted_strings)
            new_line["output"]=output


            funcname=line["func_name_dwarf"]
            new_line["funcname"]=funcname
            new_line["fid"]=fid

            json.dump(new_line,new_f)
            new_f.write("\n")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, help='Path to the input jsonl file')
    parser.add_argument('--outfile', type=str, help='Path to the save jsonl file for tokenization')
    args = parser.parse_args()

    rewrite(args)





