"""
	This file specifies the hooks that will be used to compute the gradients for the LoRA model
	It also specifies the Logger class which will be used to store the gradients and project them
"""

import torch
import torch.nn as nn


@torch.no_grad()
def compute_per_sample_gradient(
    fwd: torch.Tensor, bwd: torch.Tensor, module: nn.Module
):
    return torch.bmm(bwd.transpose(1, 2), fwd).view(bwd.shape[0], -1)

class Logger:
    def __init__(self, model_parameters, num_blocks):
        self.grads = None
        self.index = 0
        self.proj = None
        self.rank = 0
        self.corpus_data = []
        self.model_parameters = model_parameters
        # number of blocks is the number of buckets multiplied by 2 for MLP + FFN
        self.num_blocks = num_blocks

        torch.manual_seed(0)

    def init_grads(self, batch_sz, rank):
        self.grads = torch.zeros(self.num_blocks * 2, batch_sz, rank * rank).to(
            "cuda", dtype=torch.float32
        )  # start on cuda for efficiency
        self.rank = rank

    def add_datapoint(self, datapoint):
        self.corpus_data.append(datapoint)

    def hook_fn(self, module, inputs, outputs, index):
        def _grad_backward_hook_fn(grad: torch.Tensor):
            per_sample_gradient = compute_per_sample_gradient(inputs[0], grad, module)

            if per_sample_gradient.size(0) != self.grads.size(1):
                self.grads = self.grads[:,:per_sample_gradient.size(0), :]

            self.grads[index] += per_sample_gradient

        outputs.register_hook(_grad_backward_hook_fn)