from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
import gc
import torch.nn.functional as F
from math import isnan
import math
from torch.cuda.amp import  GradScaler
import json
import reasoning_datasets

import torch
from datasets import load_dataset, load_from_disk
import random
import os
import inspect
import glob
import get_model as get_model
from get_model import LinkedListCache
from accelerate import dispatch_model, infer_auto_device_map
import GPUtil
from accelerate.utils import get_balanced_memory
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import argparse

split_batch_into = 1
loss_cum = 0
eps = 2**-24


def evaluation_transformer(checkpoint_folder, ds, result_folder, batch_size = 32, gpu_memory_limit = "35GiB"):
  if not os.path.exists(checkpoint_folder):
      os.makedirs(checkpoint_folder)

  save_configs_file = os.path.join(checkpoint_folder, 'save_configs.json')
  with open(save_configs_file, 'r') as file:
      save_configs = json.load(file)

  group_size = save_configs["group_size"]
  model_name = save_configs["model_name"]
  bridges = save_configs["bridges"]
  bridges = [tuple(bridge) for bridge in bridges]
  
  if model_name == "meta-llama/Llama-3.2-1B":
    precision = torch.float32
  elif model_name == "meta-llama/Llama-3.1-8B":
    precision = torch.float16

  real_batch_size = batch_size/split_batch_into



  train_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
  # print dataloader size
  print(len(train_dataloader))

  model, tokenizer = get_model.get_model(improved = True, dataType = precision, bridges = bridges, r = 140, model_name = model_name)

  prepared_model, _ = get_model.get_model(improved = False, dataType = precision, model_name = model_name)
  llamamodel = prepared_model.model
  model.model.load_state_dict(llamamodel.state_dict(), strict = False)  
  model.lm_head = prepared_model.lm_head
  del prepared_model
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"
  tokenizer.truncation_side = "right"

  model.post_init()

  checkpoint_files = glob.glob(os.path.join(checkpoint_folder, 'pretrain_improved_*.pth'))

  iteration_numbers = []
  for file in checkpoint_files:
      filename = os.path.basename(file)
      try:
          iteration_str = filename.replace('pretrain_improved_', '').replace('.pth', '')
          iteration_numbers.append(iteration_str)
      except ValueError:
          pass


  # Specify any custom classes that shouldn't be split (if any)
  if True: #load to GPU
      no_split_modules = ["LlamaDecoderLayerLora"]
      # max_memory = {i: "75GiB" for i in range(torch.cuda.device_count())}
      max_memory = {i: gpu_memory_limit for i in range(torch.cuda.device_count())}
      max_memory = get_balanced_memory(
                          model,
                          dtype=precision,
                          low_zero=False,
                          max_memory=max_memory,
                          no_split_module_classes=no_split_modules,
                      )
      print(max_memory)
      device_map = infer_auto_device_map(
          model,
          max_memory=max_memory,
          no_split_module_classes=no_split_modules,
          dtype=precision
      )
      print(device_map)
      model.tie_weights()
      model = dispatch_model(
          model,
          device_map=device_map,
          offload_buffers=False,  # Set to True if you want to offload activations to CPU
          main_device=None,       # Specify if you have a preference for the main device
          skip_keys = model._skip_keys_device_placement
      )
      model.eval()

  for name, param in model.named_parameters():
      param.requires_grad = True

  j = 0

  result_file_name = f"{result_folder}/{model_name.split('/')[-1]}_transformer_results.json"
  if os.path.exists(result_file_name):
      with open(result_file_name, "r") as f:
          results = json.load(f)
  else:
      results = {}

  for iteration_number in iteration_numbers:
    prob_to_log = []
    if results.get(iteration_number) is not None:
      continue
    model_checkpoint = os.path.join(checkpoint_folder, f'pretrain_improved_{iteration_number}.pth')
    model.load_state_dict(torch.load(model_checkpoint, map_location='cpu'), strict = False)


    with torch.no_grad():
      for batch_idx, batch in enumerate(train_dataloader):
        perplexity_to_log = 0
        processed_batch = []

        # Handling Texts
        instruction_batch = batch['instruction']
        output_batch = batch['output']
        input_batch = batch['input']

        query_batch = ["Question: "+ instruction + "\nInput: " + input + "\n" + "Answer" for instruction, input in zip(instruction_batch, input_batch)]
        answer_batch = [": " + output for output in output_batch]
        processed_batch = [q + a for q, a in zip(query_batch, answer_batch)]

        max_length = 0
        for original_text in processed_batch:
            # Tokenize the string
            tokenized = tokenizer(original_text, padding=False, truncation=False, return_tensors='pt', add_special_tokens = False)

            # Get the token IDs
            tokens = tokenized['input_ids'][0]
            num_tokens = len(tokens)

            # If the number of tokens exceeds 1024, clip at a random position
            if num_tokens > max_length:
                max_length = num_tokens

        tokenized = tokenizer(processed_batch, padding="max_length", truncation=True, return_tensors="pt", max_length =  -(-max_length // group_size) * group_size, add_special_tokens = False)
        attention_mask = tokenized['attention_mask'].to("cuda")
        length_of_gt_answer = []
        for answer in processed_batch:
          length_of_gt_answer.append(len(tokenizer(answer, truncation=True, max_length = max_length, add_special_tokens = False)['input_ids']))
        length_of_query = []
        # for query in query_batch:
        #     length_of_query.append(len(tokenizer(query)['input_ids']))
        # We instead use decoding based method to decide the length of the query. Decode tokenized and find last occurence of "Answer:"
        for ids in tokenized['input_ids']:
          for i in range(max_length):
            decoded_text = tokenizer.decode(ids[i:])
            if "Answer:" not in decoded_text:
              length_of_query.append(i)
              break
          ## i would be the position of ":"

        

        
        for micro in range(split_batch_into):
          s = int(micro * real_batch_size)
          e = int(s + real_batch_size)
          mb_input_ids   = tokenized["input_ids"][s:e]              # <<<
          mb_attn_mask   = attention_mask[s:e]      
          for i in range(len(bridges)):
            model.model.connections[i] = None
          with torch.autocast(device_type='cuda', dtype=torch.float16):
            mb_input_ids = mb_input_ids.to("cuda")
            

            # Calculate position ids based on the attention mask
            # position_ids = torch.zeros_like(attention_mask)
            # for i, row in enumerate(attention_mask):
            #     seq_length = row.sum().item()  # Number of tokens in the sequence
            #     position_ids[i, -seq_length:] = torch.arange(0, seq_length)
            # position_ids.to("cuda")
            # position_id_this_turn = position_ids[:, 0:1]
            outputs = model(input_ids = mb_input_ids, attention_mask = mb_attn_mask, past_key_values = LinkedListCache())
            all_logits = outputs.logits
            ## Getting Prob for y|z
            all_probs = F.softmax(all_logits, dim=-1)
            probs = torch.gather(all_probs[:,:-1,:], 2, mb_input_ids[:,1:].unsqueeze(2)).squeeze()
            
            for row, length_to_consider, start_of_answer in zip(probs, length_of_gt_answer[s:e], length_of_query[s:e]):
              # Sum the elements from the start index to the end of the row
              prob_to_log.append(torch.prod(row[start_of_answer-1:length_to_consider-1].detach()).item())          # Append the result to the sum_results list
            
          ## Back prop        
          del all_logits
          del all_probs
          del probs
          del outputs
          torch.cuda.empty_cache()
          gc.collect()
          torch.cuda.empty_cache()
          gc.collect()
        j = j + 1
    # print(prob_to_log)
    results[iteration_number] = sum(prob_to_log)/len(prob_to_log)
    with open(result_file_name, "w") as f:
      json.dump(results, f)
      
if __name__ == "__main__":
  # parse args including test or valid, checkpoint folder, dataset name, batch size
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", type=bool, default=True, help="Using test or valid dataset")
  parser.add_argument("--checkpoint_folder", type=str, help="Checkpoint folder (printed in the training script)")
  parser.add_argument("--dataset_name", type=str, default="gsm8k", help="Dataset name, options: gsm8k, multi-step-arithmetic, parity")
  parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation, use a smaller batch size if you have memory issues")
  parser.add_argument("--result_folder", type=str, default="results", help="Result folder to output the json files with accuracy for each checkpointed iteration")
  parser.add_argument("--gpu_memory_limit", type=str, default="35GiB", help="GPU memory limit per device, e.g. 35GiB, 75GiB")
  args = parser.parse_args()

  test = args.test
  checkpoint_folder = args.checkpoint_folder
  dataset_name = args.dataset_name
  batch_size = args.batch_size
  result_folder = args.result_folder
  gpu_memory_limit = args.gpu_memory_limit
  if test:
    seed = 1453
  else:
    seed = 2025
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  if dataset_name == "gsm8k":
    if test:
      ds = reasoning_datasets.QAPairDataset("test.txt")
    else:
      ds = reasoning_datasets.QAPairDataset("valid.txt")
  elif dataset_name == "multi-step-arithmetic":
    ds = reasoning_datasets.RandomPruferExpressionDataset(size = 1000, max_n = 30)
  elif dataset_name == "parity":
    ds = reasoning_datasets.ParityDataset(length = 1000, seq_len = 70)
  else:
    raise ValueError(f"Dataset name {dataset_name} not supported")
  evaluation_transformer(checkpoint_folder, ds, result_folder, batch_size = batch_size, gpu_memory_limit = gpu_memory_limit)
