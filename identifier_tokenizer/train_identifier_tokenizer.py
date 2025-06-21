from pathlib import Path
from tokenizers import ByteLevelBPETokenizer,Tokenizer
import argparse
import os
import json

def train_tokenizer(args):
    tokenizer_path =os.path.join( args.vocab_out,"identifier_tokenizer.json")
    paths = [str(x) for x in Path(args.train_data).glob("**/*.txt")]
    
    tokenizer = ByteLevelBPETokenizer(add_prefix_space=True) 
    tokenizer.train(files=paths,
                    vocab_size=args.vocab_size,
                    min_frequency=args.min_frequency,
                   )
    os.makedirs(args.vocab_out, exist_ok=True)
    tokenizer.save_model(args.vocab_out)
    tokenizer.save(tokenizer_path)  


token_freq=dict()

def test_tokenizer():

    tokenizer_path = os.path.join(args.vocab_out, "identifier_tokenizer.json")
    #tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=1024)  #
    tokenizer = Tokenizer.from_file(tokenizer_path)
 
    with open(os.path.join(args.train_data,"vocab_list.txt"),"r") as f:
        for word in f:
            #print("word:",word)
            output = tokenizer.encode(word)
            #print("Normal Tokens:", output.tokens,"Normal Token Ids:", output.ids)
            #sys.exit()
            for token in output.tokens:
                token = token.replace("臓", "").replace('Ġ',"")
                if token !="":
                    if token not in token_freq.keys():
                        token_freq[token]=1
                    else:
                        token_freq[token]+=1
   
    sorted_dict_desc = dict(sorted(token_freq.items(), key=lambda item: item[1], reverse=True))  # 降序排列
    id2token, token2id = {}, {}  
    i=0
    for token, freq in sorted_dict_desc.items():  
        id2token[i] = token
        token2id[token] = i  
        i+=1


    with open(os.path.join(args.vocab_out, 'id2token.json'), 'w', encoding='utf-8') as w:
        w.write(json.dumps(id2token))
    with open(os.path.join(args.vocab_out, 'token2id.json'), 'w', encoding='utf-8') as w:
        w.write(json.dumps(token2id))



if __name__ == "__main__":
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, help='path to the input text files')
    parser.add_argument('--vocab_size', type=int, help='size of tokenizer vocabulary')
    parser.add_argument('--min_frequency', type=int, help='minimum frequency')
    parser.add_argument('--output_path', type=str, help='path to the output text files')

    args = parser.parse_args()
    tokenizer=train_tokenizer(args)
    test_tokenizer()

