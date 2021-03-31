import torch
from typing import Dict



def evaluate(
    args,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict:
    model.eval()

    correct = 0
    total = 0

    softmax = torch.nn.Softmax(dim=1)


    with torch.no_grad():
    

        for data in dataloader:
            inputs, labels = data
            #print(inputs.size())
            tmp_batch_size = len(labels)
            labels = labels.unsqueeze(1)
            # ===================forward=====================
            
            output = model(inputs.unsqueeze(1).to(device) if args.input!="CIFAR10" else inputs.to(device))
            
            out_softmax = softmax(output)

            _, predicted = torch.max(out_softmax.data, 1)
            total += tmp_batch_size

            labels = labels.view(tmp_batch_size)
            pred_cpu = predicted.cpu()
            correct += (pred_cpu == labels).sum().item()

    metrics = {"acc":correct/total}
    return metrics