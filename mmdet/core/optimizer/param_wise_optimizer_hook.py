# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict
from itertools import chain

from torch.nn.utils import clip_grad
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class ParamWiseOptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        # filter out the parameter that needs gradients
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))

        
        
        # create parameter group
        
        # filter the parameters base on the key
        
        # 
        
        # for each parameter group, apply different lr
        
        if len(params) > 0:
            # grad_returned is a value:tensor(294.0063, device='cuda:1') should be the norm over all parameters
            grad_returned = clip_grad.clip_grad_norm_(params, **self.grad_clip)
            #print('grad_returned', grad_returned)
            return grad_returned

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        #needed_para = []
        if self.grad_clip is not None:
            ### for testing
            # for name, module in runner.model.named_modules():
            #     for key, value in module.named_parameters(recurse=False):
            #         if "encoder" in name:
            #             needed_para.append(value)
            # grad_norm = self.clip_grads(needed_para)
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
            #print('grad_norm', grad_norm)
        runner.optimizer.step()
