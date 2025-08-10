from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
import gc
import torch.nn.functional as F
from math import isnan
import math
from torch.cuda.amp import  GradScaler
import json
import reasoning_datasets
import argparse
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

def train(improved, ds, checkpoint_folder, model_name, batch_size = 32, split_batch_into = 1, group_size = 8, save_interval = 3000, total_epochs = 7, multiplier = 100, gpu_memory_limit = "35GiB"):
 # Set the number of epochs
  base_lr = 3e-5
  warmup_steps = 100 # 20
  period = 900 # 80
  
  seed = 42

  if model_name == "meta-llama/Llama-3.2-1B":
    bridges = []
    for i in range(0,16,2):
        for j in range(0,16,2):
            if i - j > 5:
                bridges.append((i,j))
    precision = torch.float32
  elif model_name == "meta-llama/Llama-3.1-8B":
    bridges = []
    for i in range(6,31+1,2):
        for j in range(7,26,2):
            if i - j > 6:
                bridges.append((i,j))
    precision = torch.float16
  if not improved:
    bridges = []

  print(len(bridges))



  if not os.path.exists(checkpoint_folder):
      os.makedirs(checkpoint_folder, exist_ok=True)

  save_configs = {
    "batch_size": batch_size,
    "group_size": group_size,
    "base_lr": base_lr,
    "warmup_steps": warmup_steps,
    "period": period,
    "seed": seed,
    "bridges": bridges,
    "model_name": model_name,
    "multiplier": multiplier
  }
  # save config to checkpoint folder

  save_configs_file = os.path.join(checkpoint_folder, 'save_configs.json')
  with open(save_configs_file, 'w') as file:
      json.dump(save_configs, file)


  clip_value = batch_size
  real_batch_size = batch_size/split_batch_into
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)



  train_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)
  # print dataloader size
  print(len(train_dataloader))
  loss_cum = 0
  eps = 2**-24

  if improved:
    model, tokenizer = get_model.get_model(improved = True, dataType = precision, bridges = bridges, r = 120, model_name = model_name, multiplier = multiplier)
  else:
    model, tokenizer = get_model.get_model(improved = True, dataType = precision, bridges = bridges, r = 140, model_name = model_name)

  prepared_model, _ = get_model.get_model(improved = False, dataType = precision, model_name = model_name)
  llamamodel = prepared_model.model
  model.model.load_state_dict(llamamodel.state_dict(), strict = False)  
  model.lm_head = prepared_model.lm_head
  del prepared_model

  model.post_init()

  checkpoint_files = glob.glob(os.path.join(checkpoint_folder, 'pretrain_improved_*.pth'))

  iteration_numbers = []
  for file in checkpoint_files:
      filename = os.path.basename(file)
      try:
          iteration_str = filename.replace('pretrain_improved_', '').replace('.pth', '')
          iteration = int(iteration_str)
          iteration_numbers.append(iteration)
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
      model.train()

  for name, param in model.named_parameters():
      param.requires_grad = True

  # optimizer = Adam(model.parameters(), lr = base_lr) # 3e-5 Adam 8e-5 for 32 batch size
  optimizer = Adam([param for name, param in model.named_parameters() if 'bridge_up_proj_layers' in name or 'bridge_down_proj_layer' in name or 'lora' in name], lr = base_lr)

  loss_tracking = []
  j = 0

  if iteration_numbers:
      # Load the latest checkpoint
      latest_iteration = max(iteration_numbers)
      model_checkpoint = os.path.join(checkpoint_folder, f'pretrain_improved_{latest_iteration}.pth')
      optimizer_checkpoint = os.path.join(checkpoint_folder, f'optimizer_state_{latest_iteration}.pth')
      loss_tracking_file = os.path.join(checkpoint_folder, f'loss_tracking_{latest_iteration}.json')
      training_state_file = os.path.join(checkpoint_folder, f'training_state_{latest_iteration}.json')
      
      model.load_state_dict(torch.load(model_checkpoint, map_location='cpu'), strict = False)
      optimizer.load_state_dict(torch.load(optimizer_checkpoint, map_location='cpu'))
      with open(loss_tracking_file, 'r') as file:
          loss_tracking = json.load(file)
      with open(training_state_file, 'r') as file:
          training_state = json.load(file)
          j = training_state['iteration']
          start_epoch = training_state['epoch']  # Start from the next epoch
      print(f"Resuming training from iteration {j}, epoch {start_epoch}")
      ## place to GPU
      if True:
        model.tie_weights()
        model = dispatch_model(
            model,
            device_map=device_map,
            offload_buffers=False,  # Set to True if you want to offload activations to CPU
            main_device=None,       # Specify if you have a preference for the main device
            skip_keys = model._skip_keys_device_placement
        )
        model.train()

  else:
      # No checkpoints found; start from scratch
      j = 0
      start_epoch = 0

  optimizer_parameters = []
  for param_group in optimizer.param_groups:
      optimizer_parameters.extend(param_group['params'])

  scaler = GradScaler()

  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"
  tokenizer.truncation_side = "right"

  start_batch_number = j - start_epoch * len(train_dataloader)
  for epoch in range(start_epoch, total_epochs):
    for batch_idx, batch in enumerate(train_dataloader):
      if len(batch["input"]) < batch_size:
        continue
      batch_loss = 0
      perplexity_to_log = 0
      if (epoch == start_epoch) and batch_idx < start_batch_number:
        print(batch_idx)
        continue
      wj = j%(period + warmup_steps)
      if wj <= warmup_steps:
      # Linear warmup from 0 to base_lr
        warmup_lr = base_lr * (wj / float(warmup_steps))
        for param_group in optimizer.param_groups:
          param_group['lr'] = warmup_lr
      else:
        steps_into_decay = wj - warmup_steps
        fraction = (float(steps_into_decay))/period
        cosine_lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * fraction))
        for param_group in optimizer.param_groups:
          param_group['lr'] = cosine_lr

      optimizer.zero_grad()
        # Process each string in the batch
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

      

      prob_to_log = 0
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
          if improved:
            outputs = model(input_ids = mb_input_ids[:, 0:group_size], attention_mask = mb_attn_mask[:,:group_size], use_cache = True, past_key_values = LinkedListCache())
            all_logits = outputs.logits
            clip_length = -(-max_length // group_size) * group_size
            for i in range(group_size, clip_length, group_size): #100
              # position_id_this_turn = position_ids[:, i:i+1]
              #with torch.cuda.amp.autocast():
              outputs = model(input_ids = mb_input_ids[:, i:i+group_size], attention_mask = mb_attn_mask[:,:i+group_size], use_cache = True, past_key_values = outputs.past_key_values)
              logits = outputs.logits
              all_logits = torch.cat((all_logits, logits), dim=1)
          else:
            outputs = model(input_ids = mb_input_ids, attention_mask = mb_attn_mask, past_key_values = LinkedListCache())
            all_logits = outputs.logits

          ## Getting Prob for y|z
          all_probs = F.softmax(all_logits, dim=-1)
          probs = torch.gather(all_probs[:,:-1,:], 2, mb_input_ids[:,1:].unsqueeze(2)).squeeze()
          prod_results = []
          
          for row, length_to_consider, start_of_answer in zip(probs, length_of_gt_answer[s:e], length_of_query[s:e]):
            # Sum the elements from the start index to the end of the row
            row_prod = torch.log(row[start_of_answer-1:length_to_consider-1]+eps).sum()
            prob_to_log = prob_to_log + torch.prod(row[start_of_answer-1:length_to_consider-1].detach()).item()/batch_size          # Append the result to the sum_results list
            prod_results.append(row_prod)
          
        ## Back prop
          loss = -sum(prod_results)
          if (loss > 1000000) or isnan(loss):
            break
        
        batch_loss += loss.detach()
        scaler.scale(loss).backward(inputs=optimizer_parameters)
        if improved:
          del logits
        del all_logits
        del all_probs
        del prod_results
        del probs
        del loss
        del row_prod
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
      loss_tracking.append(prob_to_log)
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(optimizer_parameters, clip_value)
      scaler.step(optimizer)
      scaler.update()
      loss_cum = loss_cum + batch_loss
      j = j + 1
      print(f"Epoch {epoch}, Iteration {j}, Loss: {batch_loss}")
      # if j % 5 == 0:
      #   break
      if j % save_interval == 0:
        # Save model state
        full_state_dict = model.state_dict()
        selective_state_dict = {k: v for k, v in full_state_dict.items() if 'bridge_up_proj_layers' in k or 'bridge_down_proj_layer' in k or 'lora' in k}
        torch.save(selective_state_dict, os.path.join(checkpoint_folder, f'pretrain_improved_{j}.pth'))
        # Save optimizer state
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_folder, f'optimizer_state_{j}.pth'))
        # Save loss tracking data
        with open(os.path.join(checkpoint_folder, f'loss_tracking_{j}.json'), 'w') as file:
            json.dump(loss_tracking, file)
        # Save training state (iteration and epoch)
        training_state = {'iteration': j, 'epoch': epoch}
        with open(os.path.join(checkpoint_folder, f'training_state_{j}.json'), 'w') as file:
            json.dump(training_state, file)
        print(f"Checkpoint saved at iteration {j}")
      #break
    # Save checkpoint at the end of each epoch
    # torch.save(model.state_dict(), os.path.join(checkpoint_folder, f'pretrain_improved_{j}.pth'))
    # torch.save(optimizer.state_dict(), os.path.join(checkpoint_folder, f'optimizer_state_{j}.pth'))
    # with open(os.path.join(checkpoint_folder, f'loss_tracking_{j}.json'), 'w') as file:
    #     json.dump(loss_tracking, file)
    # training_state = {'iteration': j, 'epoch': epoch}
    # with open(os.path.join(checkpoint_folder, f'training_state_{j}.json'), 'w') as file:
    #     json.dump(training_state, file)
    # print(f"Checkpoint saved at the end of epoch {epoch}")

  print("Training completed.")

if __name__ == "__main__":
  # add argument parser for our method or not, model name, dataset name, checkpoint folder, batch size, group size, split batch into, save interval, total epochs
  parser = argparse.ArgumentParser()
  parser.add_argument("--our_method", type=bool, default=True, help="Whether to use our method or not")
  parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B", help="Only support Llama models for now")
  parser.add_argument("--dataset_name", type=str, default="multi-step-arithmetic", help="Dataset name, options: gsm8k, multi-step-arithmetic, parity")
  parser.add_argument("--checkpoint_root_folder", type=str, help="Checkpoint root folder. This will be used to save the checkpoints.")
  parser.add_argument("--batch_size", type=int, default=64, help="Effective batch size")
  parser.add_argument("--group_size", type=int, default=4)
  parser.add_argument("--split_batch_into", type=int, default=1, help="Splitting whole batch across different steps to reduce memory usage. Should divide the batch size.")
  parser.add_argument("--save_interval", type=int, default=1000, help="Save interval")
  parser.add_argument("--total_epochs", type=int, default=3, help="Total number of epochs")
  parser.add_argument("--multiplier", type=int, default=100, help="Multiplier parameter as in the paper. For transformer, this is not used.")
  parser.add_argument("--gpu_memory_limit", type=str, default="35GiB", help="GPU memory limit per device, e.g. 35GiB, 75GiB")
  args = parser.parse_args()

  improved = args.our_method
  model_name = args.model_name
  dataset_name = args.dataset_name
  checkpoint_root_folder = args.checkpoint_root_folder
  group_size = args.group_size
  batch_size = args.batch_size
  split_batch_into = args.split_batch_into
  save_interval = args.save_interval
  total_epochs = args.total_epochs
  multiplier = args.multiplier
  gpu_memory_limit = args.gpu_memory_limit
  model_last_name = model_name.split('/')[-1]
  if improved:
    checkpoint_folder = f'{checkpoint_root_folder}/improved_training_checkpoint_test_{model_last_name}_{dataset_name}'
  else:
    checkpoint_folder = f'{checkpoint_root_folder}/transformer_training_checkpoint_test_{model_last_name}_{dataset_name}'
  print("Checkpoint folder: " + checkpoint_folder)
  seed = 42
  random.seed(seed)

  if dataset_name == "gsm8k":
    ds = reasoning_datasets.QAPairDataset("train_shuffled.txt")
  elif dataset_name == "parity":
    ds = reasoning_datasets.ParityDataset(length = 384000, seq_len = 70)
  elif dataset_name == "multi-step-arithmetic":
    ds = reasoning_datasets.RandomPruferExpressionDataset(size = 384000, max_n = 30)

    
  train(improved, ds, checkpoint_folder, model_name, batch_size = batch_size, group_size = group_size, split_batch_into = split_batch_into, save_interval = save_interval, total_epochs = total_epochs, multiplier = multiplier, gpu_memory_limit = gpu_memory_limit)
  print("Training completed. Checkpoint saved at " + checkpoint_folder)