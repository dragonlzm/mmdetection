# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict
from itertools import chain

from torch.nn.utils import clip_grad
from mmcv.runner.hook import HOOKS, Hook


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
            grad_returned = clip_grad.clip_grad_norm_(params, **self.grad_clip)
            print('grad_returned', grad_returned)
            return grad_returned

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()
