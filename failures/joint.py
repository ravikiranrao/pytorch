import torch
import torch._dynamo
from functorch.compile import aot_function, aot_module, draw_graph, nop, print_compile  
import logging

# torch._dynamo.config.log_level = logging.DEBUG
torch.manual_seed(0)
class Mock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
    
    def forward(self, x):
        return self.linear(x)
    
mod = Mock()

# 1) Let AOT Autograd trace it. AOT Autograd can't trace .backward()
torch._dynamo.allow_in_graph(torch.autograd.grad)
# torch._dynamo.allow_in_graph(torch.autograd.backward)

def joint(x):
    out = mod(x)
    loss = out.sum()
    params = list(dict(mod.named_parameters()).values())
    grad_x, grad_w = torch.autograd.grad(loss, [x, params[0]], retain_graph=True)
    return grad_x, grad_w

x = torch.randn(2, 2, requires_grad=True)
ref = joint(x)
print("Eager Done", flush=True)

opt_fn = torch.compile(joint, backend="aot_eager")
# opt_fn = aot_function(joint, nop)
res = opt_fn(x)
# print(res)
for a, b in zip(ref, res):
    assert torch.allclose(a, b)