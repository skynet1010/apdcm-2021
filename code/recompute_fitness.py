from argparse import Namespace
from os import path, makedirs
from typing import Dict
import torch

from exec.eval import evaluate
from utils.dataloader_provider import get_test_dataloader,compute_global_mean_std
from utils.phenotype import build_architecture



def eval_fitness(args:Namespace):  
    
    d = path.join("..","models",args.input.lower())
    
    if not path.isdir(d):
        makedirs(d)
    
    model, fp = build_architecture(args.arch)
    
    best_ckt_path = path.join(d,args.filename)
    best_model_state = torch.load(best_ckt_path)
    model.load_state_dict(best_model_state["model_state_dict"])      
    
    compute_global_mean_std(args)
    test_metrics=None
    nr_of_trials = 20
    for i in range(nr_of_trials):
        print(f"#Test {i+1}/{nr_of_trials}")
        temp_test_metrics = calc_test_metrics(args,model)
        if i==0:
            test_metrics = temp_test_metrics
        else:
            test_metrics = {k: test_metrics.get(k, 0) + temp_test_metrics.get(k, 0) for k in set(test_metrics) & set(temp_test_metrics)}
    test_metrics = {k: test_metrics.get(k, 0)/nr_of_trials for k in set(test_metrics)}
    print("{}:\nFree parameters: {}\nAverage accuracy: {:.2f}%\n\n".format(args.input,fp,test_metrics['acc']*100))
    return True

def calc_test_metrics(args,model):
    test_dataloader = get_test_dataloader(args)
    test_metrics = evaluate(args,model,test_dataloader)
    return test_metrics