# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs

from .modeling_resnet import ResNetV2

from .DiffJPEG import DiffJPEG

QF_INIT = 99

# Packet_Loss_Rate = 50

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class EnhancedChannelFilter_manual(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(EnhancedChannelFilter_manual, self).__init__()
        self.num_channels = num_channels
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )
        self.threshold = nn.Parameter(torch.tensor([0.5]))
        
        # New layers to handle packet-lost data
        self.lost_packet_detection = nn.Conv2d(num_channels, num_channels, kernel_size=1, padding=0, bias=False)
        self.lost_packet_reconstruction = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=1, padding=0, bias=False)
        )
        
        # Adaptive parameters for loss rate
        self.packet_loss_adaptation = nn.Linear(1, num_channels, bias=False)  # Learnable layer for loss rate adaptation

        # YOTO algorithm parameters
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Initial learning rate for YOTO
        self.beta = nn.Parameter(torch.tensor(0.9))   # Momentum for YOTO        

    def forward(self, x, Packet_Loss_Rate):
        B, C, _, _ = x.size()
        
        # Simulate packet loss
        # x = self.packet_loss_to_partial_patch(x, Packet_Loss_Rate)
        x = packet_loss_to_partial_patch(x, Packet_Loss_Rate)
        loss_x = x
        
        # Global average pooling and importance scoring
        y = self.global_avg_pool(x).view(B, C)
        importance_scores = self.fc(y).view(B, C, 1, 1)

        # Adaptation based on packet loss rate
        loss_rate_tensor = Packet_Loss_Rate * torch.ones((B, 1), device=x.device, dtype=x.dtype)
        loss_adaptation = self.packet_loss_adaptation(loss_rate_tensor).view(B, C, 1, 1)
        mask = torch.relu(importance_scores + loss_adaptation - self.threshold)
        
        # Detect lost packets
        lost_packet_mask = self.lost_packet_detection(x)
        sigmoid_detected_x = torch.sigmoid(lost_packet_mask)
        
        # Reconstruct lost packets
        reconstructed_x = self.lost_packet_reconstruction(torch.cat([x, sigmoid_detected_x * x], dim=1))
        
        # Apply channel importance filter
        filtered_x = reconstructed_x * mask

        # Update YOTO parameters based on Packet_Loss_Rate
        self.alpha.data = self.alpha.data * (1 - Packet_Loss_Rate / 100)  # Divide by 100 to convert percentage to fraction
        self.beta.data = self.beta.data * (1 - Packet_Loss_Rate / 100)
        
        return filtered_x



class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        print(f"\n in_channels: {in_channels} n_patches: {n_patches} config.hidden_size: {config.hidden_size} patch_size: {patch_size} ")
        
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

        # self.jpeg_compression_patch = JpegCompression_patch()

        # # Assuming patch_size and other parameters are defined
        # self.patch_selector = PatchSelector(num_patches=768, embed_dim=config.hidden_size)

        self.enhanced_channel_filter = EnhancedChannelFilter_manual(config.hidden_size)
        


        # Initialize PatchSelector
        
        self.effect_threshold = 1e-5  # Define a threshold for channel effectiveness
        
        
    # def forward(self, x):
    def forward(self, x, Packet_Loss_Rate):
        print(f"\n x shape: {x.shape} \n")
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)

        print(f"\n x shape after patch_embeddings: {x.shape} \n")

        

        # Assuming x has already passed through self.patch_embeddings
        x_before = x.clone()  # Clone x to keep its pre-filter state
        
        # Apply channel-wise filtering
        # x_after = self.channel_filter(x)
        # print(f"\nPacket_Loss_Rate: {Packet_Loss_Rate}\n")
        x_after = self.enhanced_channel_filter(x, Packet_Loss_Rate)
        
        # Compute the mean absolute value for each channel before and after filtering
        # abs mean value for each channel
        # make copy of x_before and x_after
        mean_abs_before = x_before.abs().mean(dim=[0, 2, 3])  # Shape: [C]
        mean_abs_after = x_after.abs().mean(dim=[0, 2, 3])  # Shape: [C]

        # mean_abs_before = x_before.mean(dim=[0, 2, 3])  # Shape: [C]
        # mean_abs_after = x_after.mean(dim=[0, 2, 3])  # Shape: [C]
        
        # Determine the threshold for considering a channel "close to zero"
        threshold = 0.01  # This threshold can be adjusted
        
        # Identify channels that are "active" before and "close to zero" after
        active_before = mean_abs_before > threshold
        close_to_zero_after = mean_abs_after <= threshold
        channels_closer_to_zero = (active_before & close_to_zero_after)
        
        num_channels_closer_to_zero = channels_closer_to_zero.sum().item()
        channels_closer_to_zero_indices = channels_closer_to_zero.nonzero(as_tuple=False).squeeze().tolist()

        # print(channels_closer_to_zero.shape)

        # print(f"\nNumber of channels out of 768 closer to zero after filtering: {num_channels_closer_to_zero}\n")

        # print(f"Indices of channels closer to zero: {channels_closer_to_zero_indices}\n")
        # print("\n",x_before[:,5,:,:], "\n")
        # print("\n",x_after[:,5,:,:], "\n")
        # print("\n",x_after[:,9,:,:], "\n")
        # print("\n",x_after[:,22,:,:], "\n")        
        
        x = x_after

        x = x.flatten(2)

        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings




def packet_loss_to_partial_patch(x, loss_rate=50):
    assert x.ndim == 4, "Input tensor must have 4 dimensions (batch, channel, height, width)"
    
    # Clone the original tensor for later comparison
    x_clone = x.clone()
    
    # Flatten the tensor following the specified order: Batches, then height and width elements across channels
    x_permuted = x.permute(0, 2, 3, 1).flatten()
    
    # Define the UDP packet characteristics
    chunk_size = 1472  # The size of each chunk in bytes
    total_bytes = x_permuted.numel() * x_permuted.element_size()  # Total number of bytes in the tensor
    num_chunks = (total_bytes + chunk_size - 1) // chunk_size  # Number of chunks

    # Calculate the number of chunks to zero out based on the loss rate
    num_lossy_chunks = int(np.ceil(num_chunks * (loss_rate / 100)))
    
    # Create a list to hold each chunk
    udp_chunks = []
    
    # Split the data into chunks
    for i in range(num_chunks):
        start_index = i * chunk_size // x_permuted.element_size()
        end_index = min((i + 1) * chunk_size // x_permuted.element_size(), x_permuted.numel())
        udp_chunk = x_permuted[start_index:end_index]
        udp_chunks.append(udp_chunk)
    
    # Randomly select chunks to simulate packet loss
    loss_indices = np.random.choice(range(num_chunks), size=num_lossy_chunks, replace=False)
    
    # Simulate packet loss by zeroing out selected chunks
    for index in loss_indices:
        udp_chunks[index].zero_()

    # Reconstruct the data from chunks
    reconstructed_data = torch.cat([chunk for chunk in udp_chunks])
    
    # Reshape back to the original permuted dimensions, then de-permute to original shape
    reconstructed_tensor = reconstructed_data.view_as(x_permuted).view(x.size(0), x.size(2), x.size(3), x.size(1)).permute(0, 3, 1, 2)
    
    # Calculate and print the percentage of zeros
    zeros_in_original = x_clone.numel() - x_clone.nonzero().size(0)
    zeros_in_reconstructed = reconstructed_tensor.numel() - reconstructed_tensor.nonzero().size(0)
    percentage_zeros = 100 * (zeros_in_reconstructed - zeros_in_original) / x_clone.numel()

    # print(f"Percentage of elements changed to zero: {percentage_zeros:.2f}%")
    # print(f"Packet loss indices: {loss_indices}")
    
    return reconstructed_tensor





class JpegCompression_patch(nn.Module):
    def __init__(self, compression_strength_init=QF_INIT):
        
        super(JpegCompression_patch, self).__init__()
        # self.compression_strength = nn.Parameter(torch.tensor([compression_strength_init], dtype=torch.float32))
        # self.compression_strength = nn.Parameter(torch.tensor([QF_INIT], dtype=torch.float32))
        self.compression_strength = torch.tensor([QF_INIT], dtype=torch.float32).cuda()
        print("\ncompression_strength_init", self.compression_strength, "\n")


    def forward(self, x):
        orig_min = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        orig_max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        # print("x: ", x.shape, x[0].mean(), x[0])
        
        # x_replicated_copy = x.clone()
        
        # Normalize to [0, 255]
        # x_normalized = (x - orig_min) / (orig_max - orig_min + 1e-5) * 255
        x_normalized = (x - orig_min) / (orig_max - orig_min + 1e-5)
        
        x_replicated_copy = x_normalized.clone()

        # batch, 768, 14, 14
        batch, channels, height, width = x_normalized.shape

        # # Replicate and pad the input tensor
        x_replicated = x_normalized.view(batch, channels, 1, height, width)
        # x_replicated = x_normalized.view(-1, 3, 16, 16)
        # x_replicated = x_normalized.reshape(-1, 3, 16, 16)
        x_replicated = x_replicated.repeat(1, 1, 3, 1, 1)
        x_replicated = x_replicated.view(batch * channels, 3, height, width)
        # # x_replicated_copy = x_replicated.clone()
        x_padded = nn.functional.pad(x_replicated, (1, 1, 1, 1), mode='replicate')
        # # print('x_padded: ', x_padded.shape)

        # Apply JPEG compression
        # JPEG = DiffJPEG(height=height, width=width, differentiable=True, quality=self.compression_strength).cuda()
        JPEG = DiffJPEG(height=16, width=16, differentiable=True, quality=self.compression_strength).cuda()
        compressed_x = JPEG(x_padded)
        # compressed_x = JPEG(x_replicated)
        # compressed_x = x_padded
        # print('compressed_x: ', compressed_x.shape)

        # # Resize back to original dimensions
        compressed_x = compressed_x[:, :, 1:-1, 1:-1]
        # # compressed_x_copy = compressed_x.clone()

        # # Reshape and average channels
        compressed_x = compressed_x.view(batch, channels, 3, height, width)
        # compressed_x = compressed_x.view(batch, channels, height, width)
        # compressed_x = compressed_x.reshape(batch, channels, height, width)
        compressed_x = compressed_x.mean(dim=2)

        # compressed_x_copy = compressed_x.clone()

        # Denormalize
        # compressed_x = (compressed_x / 255) * (orig_max - orig_min + 1e-5) + orig_min
        compressed_x = (compressed_x) * (orig_max - orig_min + 1e-5) + orig_min

        compressed_x_copy = compressed_x.clone()
        
        gap_before_after = torch.abs(x_replicated_copy - compressed_x_copy)
        # print('gap_before_after: ', gap_before_after[0].mean())
        # print('gap_before_after: ', gap_before_after[0])

        return compressed_x                




class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

        # self.jpeg_compression_attention = JpegCompression_attention()

    def transpose_for_scores(self, x):
        print(f"\n x shape in attention: {x.shape} \n")

        

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)

        print(f"\n new_x_shape: {new_x_shape} \n")

        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        print(f"\n query_layer shape: {query_layer.shape} \n")
        print(f"\n key_layer shape: {key_layer.shape} \n")
        print(f"\n value_layer shape: {value_layer.shape} \n")

        

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Applly JPEG compression to attention_scores here
        # attention_scores = self.jpeg_compression_attention(attention_scores)

        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        print(f"\n attention_output shape: {attention_output.shape} \n")



        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x




class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)

        print(f"\n encoded shape: {encoded.shape} \n")
        

        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    # def forward(self, input_ids):
    #     embedding_output = self.embeddings(input_ids)
    #     encoded, attn_weights = self.encoder(embedding_output)
    #     return encoded, attn_weights


    def forward(self, input_ids, Packet_Loss_Rate):
        embedding_output = self.embeddings(input_ids, Packet_Loss_Rate)  # Adjust embeddings to receive this
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights        



class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

        # self.jpeg_compression = JpegCompression()


    def forward(self, x, Packet_Loss_Rate, labels=None):
        x, attn_weights = self.transformer(x, Packet_Loss_Rate)  # Ensure transformer can accept this

        print(f"\n x after transformer shape: {x.shape} \n")

        logits = self.head(x[:, 0])

        print(f"\n logits shape: {logits.shape} \n")
        # sys.exit()

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights


    # def forward(self, x, labels=None):

    #     # Apply JPEG compression to input feature in network here
    #     # x = self.jpeg_compression(x)

    #     x, attn_weights = self.transformer(x)
    #     logits = self.head(x[:, 0])

    #     if labels is not None:
    #         loss_fct = CrossEntropyLoss()
    #         loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
    #         return loss
    #     else:
    #         return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
