from torch.nn.modules.module import Module
from torch.autograd import Function, Variable
import channelnorm_cuda

class ChannelNormFunction(Function):

    @staticmethod
    def forward(ctx, input1, norm_deg=2):
        assert input1.is_contiguous()

        ctx.save_for_backward(input1)
        ctx.norm_deg = norm_deg

        b, d, h, w = input1.size()
        output = input1.new(b, d, h, w).zero_()
        channelnorm_cuda.forward(input1, output, norm_deg)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        assert grad_output.is_contiguous()

        input1 = ctx.saved_tensors[0]

        b, d, h, w = input1.size()
        grad_input1 = Variable(input1.new(b, d, h, w).zero_())
        channelnorm_cuda.backward(input1, grad_output.data, grad_input1.data, ctx.norm_deg)

        return grad_input1, None

class ChannelNorm(Module):

    def __init__(self, norm_deg=2):
        super(ChannelNorm, self).__init__()
        self.norm_deg = norm_deg

    def forward(self, input1):
        return ChannelNormFunction.apply(input1, self.norm_deg)

