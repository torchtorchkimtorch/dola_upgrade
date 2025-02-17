import argparse
import time
import csv
import tqdm
import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria

import argparse
import warnings
import pandas as pd
import numpy as np

class DoLa:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=40):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory
        self.mode, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        if self.device == "cuda":
            config = AutoConfig.from_pretrained(model_name)
            # fix torch_dtype variable as automatical variable 
            torch_dtype = getattr(config, "torch_dtype", torch.float16)
            kwargs = {"torch_dtype":torch_dtype, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i:f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
                elif self.device == "cpu":
                    kwargs = {}
                else:
                    raise ValueError(f"Invalid device: {self.device}")
        tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
        model = AutoModelForCausalLM.from_pretrained(model_name,low_cpu_mem_usage=True,**kwargs)
        if self.device == "cuda" and self.num_gpus==1:
            model.cuda()
        return model, tokenizer

    # 여기 아직 안했음 
    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n'+stop_word)[:3]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ",stop_word, ' with the ids', stop_word_ids, flush=True)
        