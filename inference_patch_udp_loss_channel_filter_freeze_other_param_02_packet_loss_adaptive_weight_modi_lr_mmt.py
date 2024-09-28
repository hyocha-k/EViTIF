# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from models.modeling_patch_udp_loss_channel_filter_freeze_other_param_02_loss_adaptive_weight_modi_lr_mmt_inference import VisionTransformer, CONFIGS
import models.modeling_patch_udp_loss_channel_filter_freeze_other_param_02_loss_adaptive_weight_modi_lr_mmt_inference as mdl


from torch.utils.data import Subset
# from deepspeed.profiling.flops_profiler import get_model_profile



def load_model(args):
    config = CONFIGS[args.model_type]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 100 if args.dataset == 'cifar100' else 10
    model = VisionTransformer(config, img_size=args.img_size, num_classes=num_classes, zero_head=False)
    model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device), strict=False)
    model.eval()
    return model

# def prepare_data_loaders(args):
    # transform = transforms.Compose([
    #     transforms.Resize((args.img_size, args.img_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    
#     dataset_class = torchvision.datasets.CIFAR100 if args.dataset == 'cifar100' else torchvision.datasets.CIFAR10
#     testset = dataset_class(root='./data', train=False, download=True, transform=transform)

#     # testset_subset = Subset(testset, [args.batch_size])  # Use a single sample for testing

#     # Specify a range or a list of indices to include as a subset
#     # For example, to use the first 100 samples:
#     n_set = 256 / args.batch_size

#     subset_indices = list(range(args.batch_size * int(n_set)))
#     # Or, to use a specific list of indices:
#     # subset_indices = [0, 1, 2, 3, 4, 99, 100, 101, 102, 103]

#     testset_subset = Subset(testset, subset_indices)

#     testloader = torch.utils.data.DataLoader(testset_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
#     return testloader



# def prepare_data_loaders(args):
#     transform = transforms.Compose([
#         transforms.Resize((args.img_size, args.img_size)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])

#     # Load the successful inference data and labels
#     successful_images = torch.load('./inferred_data/successful_images.pt')
#     successful_labels = torch.load('./inferred_data/successful_labels.pt')
        
    
#     dataset_class = torchvision.datasets.CIFAR100 if args.dataset == 'cifar100' else torchvision.datasets.CIFAR10
#     # testset = dataset_class(root='./data', train=False, download=True, transform=transform)

#     # testset data shape:  torch.Size([239, 3, 224, 224])
#     testset = TensorDataset(successful_images, successful_labels)
    
#     # slice testset into specific batch size
#     # subset_indices = list(range(args.batch_size))
#     subset_indices = list(range(args.batch_size, args.batch_size+1))
#     testset_subset = Subset(testset, subset_indices)
#     # print("testset data shape: ", testset_subset.shape)

#     # testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
#     # testloader = torch.utils.data.DataLoader(testset_subset, batch_size=args.batch_size+1, shuffle=False, num_workers=2)
#     testloader = torch.utils.data.DataLoader(testset_subset, batch_size=1, shuffle=False, num_workers=2)    
#     return testloader




def prepare_data_loaders(args):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load the successful inference data and labels
    # successful_images = torch.load('./inferred_data/successful_images.pt')
    # successful_labels = torch.load('./inferred_data/successful_labels.pt')
        
    
    dataset_class = torchvision.datasets.CIFAR100 if args.dataset == 'cifar100' else torchvision.datasets.CIFAR10
    testset = dataset_class(root='./data', train=False, download=True, transform=transform)

    # testset data shape:  torch.Size([239, 3, 224, 224])
    # testset = TensorDataset(successful_images, successful_labels)
    
    # # slice testset into specific batch size
    # # subset_indices = list(range(args.batch_size))
    # subset_indices = list(range(args.batch_size, args.batch_size+1))
    # testset_subset = Subset(testset, subset_indices)
    # # print("testset data shape: ", testset_subset.shape)

    # testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    # testloader = torch.utils.data.DataLoader(testset_subset, batch_size=args.batch_size+1, shuffle=False, num_workers=2)
    # testloader = torch.utils.data.DataLoader(testset_subset, batch_size=1, shuffle=False, num_workers=2)   
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2) 
    return testloader    




def infer(args, model, dataloader, device):
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            # Wrap the inference inside the profiler context
            # with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
            #     outputs = model(images)
            # with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
            outputs = model(images, packet_lr)
            # outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print('\nFinal Accuracy: {:.4f} %'.format(100 * correct / total), " packet_loss_rate: ", f"{packet_lr:.4f}", "%",  f" elements changed to zero: {mdl.percentage_zeroed:.2f} %", f" patch subsample number: {patch_subsample_num:.0f} ", f" sample wise: {sample_wise} ", f" total sample number: {mdl.Total_Sample_Number:.0f} ")
    print('\nFinal Accuracy: {:.4f} %'.format(100 * correct / total), " packet_loss_rate: ", f"{packet_lr:.4f} %\n")
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))




def main():
    parser = argparse.ArgumentParser(description='CIFAR-100 Inference with Vision Transformer')
    parser.add_argument('--model_type', default='ViT-B_16', help='Model type from CONFIGS')
    parser.add_argument('--pretrained_model_path', type=str, help='Path to the pretrained model')
    parser.add_argument('--img_size', default=224, type=int, help='Input image size')
    parser.add_argument('--batch_size', default=64, type=int, help='Inference batch size')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use for inference (cifar10 or cifar100)')
    parser.add_argument("--qf_init", default=99, type=float, help="The initial qf.")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--packet_loss_rate", type=float, default=0, help="Local rank. Necessary for using the torch.distributed.launch utility.")    
    parser.add_argument("--patch_subsample_num", type=int, default=0, help="Local rank. Necessary for using the torch.distributed.launch utility.")     
    parser.add_argument("--sample_wise", type=str, default="vector", help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--discard_percent", type=float, default=0, help="Local rank. Necessary for using the torch.distributed.launch utility.")


    args = parser.parse_args()
    mdl.QF_INIT = args.qf_init
    mdl.Packet_Loss_Rate = args.packet_loss_rate
    mdl.Patch_Subsample_Num = args.patch_subsample_num
    mdl.Sample_Wise = args.sample_wise
    mdl.Discard_Percent = args.discard_percent
    global QF
    global packet_lr
    global patch_subsample_num
    global sample_wise
    sample_wise = args.sample_wise
    patch_subsample_num = args.patch_subsample_num
    packet_lr = args.packet_loss_rate
    QF = args.qf_init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_cifar10(args) if args.dataset == 'cifar10' else load_model(args)
    model.to(device)
    testloader = prepare_data_loaders(args)
    infer(args, model, testloader, device)

if __name__ == "__main__":
    main()
