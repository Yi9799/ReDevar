# Recovering Variable Names in Decompiled Code Based on Multi-task Learning

## Dataset

VarBERT：https://www.dropbox.com/scl/fo/3thmg8xoq2ugtjwjcgjsm/h?rlkey=azgjeq513g4semc1qdi5xyroj&dl=0

Resym：https://zenodo.org/records/13923982

## Requirements

python                   3.8

numpy                   1.24.3

tqdm                    4.66.5

transformers            4.46.3

pandas                  2.0.3

scikit-learn            1.3.2

pycparser               2.22

tensorboard             2.14.0

tensorboardX            2.6.2.2

tokenizers              0.20.0

torch                   2.4.1



## The directory structure and contents  are as follows

- tokenizer

  -- get_file.py # Generates corpus for training code tokenizer

  -- train_tokenizer.py # Trains code tokenizer



- identifier_tokenizer
  -- generate_vocab.py # Generates corpus (vocab_list.txt) for training the identifier tokenizer and build the vocabulary for name recovery
  
  -- train_identifier_tokenizer.py # Training the identifier tokenizer to acquire the complexity variable name 



- process
  -- data_origin.py # Extracts data sources of variable instances
  
  -- preprocess_data.py # Converts input decompiled code into model-acceptable format
  
  -- exclude_invalid.py # Filters out invalid samples from the dataset
  
  -- for_Resym.py # Converts dataset into Resym-compatible format



- approach
  -- eval_inf.py # Infer and evaluate
  
  -- model.py # Model 
  
  -- train_val.py # Train and validate
  
  -- utils.py # Includes data loading and utility functions
  
  -- resize_model.py # Adjusts model size when necessary



## To obtain results, please follow these steps in order

1. Perform data processing (run scripts under tokenizer and identifier_tokenizer directories, then run process_data.py)
2. Train the model (train_val.py)
3. Run inference and evaluation (inf_val.py)

Corresponding execution commands:

### Run scripts in identifier_tokenizer

```
python generate_vocab.py \
    --train_file <path_to_raw_train_file.jsonl> \
    --vocab_size <vocab_size> \
    --output_dir <path_to_vocabfiles> 
```



```
python train_identifier_tokenizer.py \
 --input_path <path_to_vocab_list.txt> \
 --vocab_size < len of vocab_list>\
 --min_frequency 2 \
 --output_path <path_to_identifier_tokenizer>
```



### Run scripts in tokenizer

```
python get_file.py \
--input_file <path_to_train.jsonl> \
--output_file <path_to_process_file.txt> 
```

```
python train_bpe_tokenizer.py \
 --input_path <path_to_process_file.tx> \
 --vocab_size 50265 \
 --min_frequency 2 \
 --output_path <path_to_tokenizer>
```



### Run preprocess.py to generate source and target sequences for the model

```
python preprocess.py \
--train_file <path_to_raw_train_file.jsonl> \
--test_file <path_to_raw_test_file.jsonl> \
--test_file <path_to_raw_validation_file.jsonl> \
--tokenizer <path_to_tokenizer> \
--identifier_tokenizer <path_to_identifier_tokenizer> \
--vocab_word_to_idx  <path_to_vocabfiles/word_to_idx.json> \
--vocab_idx_to_word  <path_to_vocabfiles/idx_to_word.json> \
--decompiler <ida_or_ghidra> \
--max_chunk_size 800 \
--workers 26 \
--out_train_file <path_of_preprocessed_files/preprocessed_train_src.json> \
--out_test_file <path_of_preprocessed_files/preprocessed_test_src.json> \
--out_validation_file <path_of_preprocessed_files/preprocessed_validation_src.json>


```



### Train the model

```
set CUDA_LAUNCH_BLOCKING=1 && python train_val.py \
--evaluate_during_training \
--eval_data_file <path_of_preprocessed_files/preprocessed_validation_src.json> \
 --overwrite_output_dir \
--train_data_file <path_of_preprocessed_files/preprocessed_train_src.json> \
--output_dir <path_to_save_trained_model> \
--block_size 800 \
--model_type roberta \
--model_name_or_path <path_of_cmlm_model>  \
--tokenizer_name <path_to_tokenizer> \
--vocab_path <path_to_vocabfiles/word_to_idx.json> \
--do_train \
--num_train_epochs 15\
--save_steps (len of dataloader)// args.gradient_accumulation_steps\
--logging_steps 5000 \
--per_gpu_train_batch_size 4 \
--weight_decay 0.01 \
--learning_rate 5e-5 \
--gradient_accumulation_steps 2
```



### Run inference and evaluation

```
python inf_eval.py \
--model_name <path_to_model> \
--tokenizer_name <path_to_tokenizer> \
--block_size 800 \
--data_file <path_of_preprocessed_files/preprocessed_test.json> \
--prefix test _inference \
--batch_size 4 \
--pred_path <path_to_save_inference_result> \
--out_vocab_map <path_to_vocabfiles/idx_to_word.json>
```

