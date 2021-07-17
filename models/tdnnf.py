# This implementation is based on: https://github.com/cvqluu/Factorized-TDNN

import torch
import torch.nn.functional as F
import torch.nn as nn
from .tdnn import TDNN

class DenseReLU(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(DenseReLU, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.nl = nn.ReLU()

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.transpose(1, 2)
        x = self.fc(x)
        x = self.nl(x)
        if len(x.shape) > 2:
            x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.bn(x)
        if len(x.shape) > 2:
            x = x.transpose(1, 2)
        return x

class StatsPool(nn.Module):

    def __init__(self, floor=1e-10, bessel=False):
        super(StatsPool, self).__init__()
        self.floor = floor
        self.bessel = bessel

    def forward(self, x):
        # x = [B,dim,len]
        x = x.transpose(1, 2)
        means = torch.mean(x, dim=1)
        _, t, _ = x.shape
        if self.bessel:
            t = t - 1
        residuals = x - means.unsqueeze(1)
        numerator = torch.sum(residuals**2, dim=1)
        stds = torch.sqrt(torch.clamp(numerator, min=self.floor)/t)
        x = torch.cat([means, stds], dim=1)
        return x

class SharedDimScaleDropout(nn.Module):
    def __init__(self, alpha: float = 0.5, dim=1):
        '''
        Continuous scaled dropout that is const over chosen dim (usually across time)
        Multiplies inputs by random mask taken from Uniform([1 - 2\alpha, 1 + 2\alpha])
        '''
        super(SharedDimScaleDropout, self).__init__()
        if alpha > 0.5 or alpha < 0:
            raise ValueError("alpha must be between 0 and 0.5")
        self.alpha = alpha
        self.dim = dim
        self.register_buffer('mask', torch.tensor(0.))

    def forward(self, X):
        if self.training:
            if self.alpha != 0.:
                # sample mask from uniform dist with dim of length 1 in self.dim and then repeat to match size
                tied_mask_shape = list(X.shape)
                tied_mask_shape[self.dim] = 1
                repeats = [1 if i != self.dim else X.shape[self.dim]
                           for i in range(len(X.shape))]
                return X * self.mask.repeat(tied_mask_shape).uniform_(1 - 2*self.alpha, 1 + 2*self.alpha).repeat(repeats)
                # expected value of dropout mask is 1 so no need to scale outputs like vanilla dropout
        return X


class SemiOrthogonalConv(TDNN):

    def __init__(self,
                input_dim: int,
                output_dim: int,
                context: list,
                padding: int,
                init: str = 'xavier'):
        """
        Semi-orthogonal convolutions. The forward function takes an additional
        parameter that specifies whether to take the semi-orthogonality step.
        :param context: The temporal context
        :param input_dim: The number of input channels
        :param output_dim: The number of channels produced by the temporal convolution
        :param init: Initialization method for weight matrix (default = Kaldi-style)
        """
        super(SemiOrthogonalConv, self).__init__(input_dim, output_dim, context, padding, bias=False)
        self.init_method = init
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_method == 'kaldi':
            # Standard dev of M init values is inverse of sqrt of num cols
            torch.nn.init._no_grad_normal_(
                self.temporal_conv.weight, 0.,
                self.get_M_shape(
                    self.temporal_conv.weight
                )[1]**-0.5)
        elif self.init_method == 'xavier':
            # Use Xavier initialization
            torch.nn.init.xavier_normal_(
                self.temporal_conv.weight
            )

    def step_semi_orth(self):
        with torch.no_grad():
            M = self.get_semi_orth_weight(self.temporal_conv.weight)
            self.temporal_conv.weight.copy_(M)

    @staticmethod
    def get_semi_orth_weight(M):
        """
        Update Conv1D weight by applying semi-orthogonality.
        :param M: Conv1D weight tensor
        """
        with torch.no_grad():
            update_speed = 0.125
            orig_shape = M.shape
            # a conv weight differs slightly from TDNN formulation:
            # Conv weight: (out_filters, in_filters, kernel_width)
            # TDNN weight M is of shape: (in_dim, out_dim) or [rows, cols]
            # the in_dim of the TDNN weight is equivalent to in_filters * kernel_width of the Conv
            M = M.reshape(
                orig_shape[0], orig_shape[1]*orig_shape[2]).T
            # M now has shape (in_dim[rows], out_dim[cols])
            mshape = M.shape
            if mshape[0] > mshape[1]:    # semi orthogonal constraint for rows > cols
                M = M.T
            P = torch.mm(M, M.T)
            PP = torch.mm(P, P.T)
            trace_P = torch.trace(P)
            trace_PP = torch.trace(PP)
            ratio = trace_PP * P.shape[0] / (trace_P * trace_P)

            # the following is the tweak to avoid divergence (more info in Kaldi)
            # assert ratio > 0.9, "Ratio of traces is less than 0.9"
            if ratio > 1.02:
                update_speed *= 0.5
                if ratio > 1.1:
                    update_speed *= 0.5
            scale2 = trace_PP/trace_P
            update = P - (torch.matrix_power(P, 0) * scale2)
            alpha = update_speed / scale2
            update = (-4.0 * alpha) * torch.mm(update, M)
            M_new = M + update
            # M_new has shape (cols, rows) if rows > cols, else has shape (rows, cols)
            # Transpose (or not) to shape (cols, rows) (IMPORTANT, s.t. correct dimensions are reshaped)
            # Then reshape to (cols, in_filters, kernel_width)
            return M_new.reshape(*orig_shape) if mshape[0] > mshape[1] else M_new.T.reshape(*orig_shape)

    @staticmethod
    def get_M_shape(conv_weight):
        orig_shape = conv_weight.shape
        return (orig_shape[1]*orig_shape[2], orig_shape[0])

    def orth_error(self):
        return self.get_semi_orth_error(self.temporal_conv.weight).item()
    
    @staticmethod
    def get_semi_orth_error(M):
        with torch.no_grad():
            orig_shape = M.shape
            M = M.reshape(
                orig_shape[0], orig_shape[1]*orig_shape[2]).T
            mshape = M.shape
            if mshape[0] > mshape[1]:    # semi orthogonal constraint for rows > cols
                M = M.T
            P = torch.mm(M, M.T)
            return torch.norm(P, p='fro')

    def forward(self, x, semi_ortho_step = False):
        """
        :param x: is one batch of data, x.size(): [batch_size, input_dim, sequence_length]
            sequence length is the dimension of the arbitrary length data
        :param semi_ortho_step: If True, take a step towards semi-orthogonality
        :return: [batch_size, output_dim, sequence_length - kernel_size + 1]
        """
        if semi_ortho_step:
            self.step_semi_orth()
        return self.temporal_conv(x)

                
class TDNNF_type1(torch.nn.Module):
    def __init__(self,
                input_dim: int,
                output_dim: int,
                bottleneck_dim: int,
                time_stride: int,
                padding: str = "same"):
        """
        Implementation of a factorized TDNN layer (see Povey et al., "Semi-Orthogonal 
        Low-Rank Matrix Factorization for Deep Neural Networks", Interspeech 2018).
        We implement the 3-stage splicing method, where each layer implicitly contains
        transformation from input_dim -> bottleneck_dim -> bottleneck_dim -> output_dim.
        The semi-orthogonality step is taken once every 4 iterations. Since it is hard
        to track iterations within the module, we generate a random number between 0
        and 1, and take the step if the generated number is below 0.25.
        :param input_dim: The hidden dimension of previous layer
        :param output_dim: The number of output dimensions
        :param bottleneck_dim: The dimensionality of constrained matrices
        :param time_stride: Controls the time offset in the splicing
        """
        super(TDNNF, self).__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        assert time_stride >= 3
        
        if time_stride == 0:
            context_f1 = context_f2 = context_f3 = [0]
            padding_f1 = padding_f2 = padding_f3 = 0
        else:
            assert padding in ["same","valid"]
            context_f1 = [-1*(time_stride//2+1),0]
            #context_f1 = [-1*time_stride, 0]
            context_f2 = [0,time_stride//2+1]
            #context_f2 = [0,time_stride]
            context_f3 = [0,time_stride]
            if padding == "same" and time_stride % 2 == 1:
                padding_f1 = int(time_stride) // 2 
                padding_f2 = int(time_stride) // 2 + 1
            elif padding == "same" and time_stride % 2 == 0:
                padding_f1 = int(time_stride) // 2 
                padding_f2 = int(time_stride) // 2
            elif padding == "valid":
                padding_f1 = 0
                padding_f2 = 0
        
        self.factor1 = SemiOrthogonalConv(input_dim, bottleneck_dim, context_f1, padding_f1)
        self.factor2 = SemiOrthogonalConv(bottleneck_dim, bottleneck_dim, context_f2,padding_f2)
        self.factor3 = TDNN(bottleneck_dim, output_dim, context_f3, padding_f3)
        self.nl = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.output_dim)
        self.dropout = SharedDimScaleDropout(alpha=0.5, dim=1)


    def forward(self, x, semi_ortho_step=True):
        """
        :param x: is one batch of data, x.size(): [batch_size, input_dim, in_seq_length]
            sequence length is the dimension of the arbitrary length data
        :param semi_ortho_step: if True, update parameter for semi-orthogonality
        :return: [batch_size, output_dim, out_seq_length]
        """
        x = self.factor1(x, semi_ortho_step=semi_ortho_step)
        x = self.factor2(x, semi_ortho_step=semi_ortho_step)
        x = self.factor3(x)
        
        
        return x

    def orth_error(self):
        """
        Compute semi-orthogonality error (for debugging purposes).
        """
        orth_error = 0
        for layer in [self.factor1, self.factor2]:
            orth_error += layer.orth_error()
        return orth_error


class TDNNF(torch.nn.Module):
    def __init__(self,
                input_dim: int,
                output_dim: int,
                bottleneck_dim: int,
                time_stride: int,
                padding: str = "same"):
        """
        Implementation of a factorized TDNN layer (see Povey et al., "Semi-Orthogonal 
        Low-Rank Matrix Factorization for Deep Neural Networks", Interspeech 2018).
        input_dim -> bottleneck_dim -> output_dim.
        The semi-orthogonality step is taken once every 4 iterations. Since it is hard
        to track iterations within the module, we generate a random number between 0
        and 1, and take the step if the generated number is below 0.25.
        :param input_dim: The hidden dimension of previous layer
        :param output_dim: The number of output dimensions
        :param bottleneck_dim: The dimensionality of constrained matrices
        :param time_stride: Controls the time offset in the splicing
        """
        super(TDNNF, self).__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        
        if time_stride == 0:
            context_f1 = context_f2 = [0]
            padding_f1 = padding_f2 = 0
        else:
            assert padding in ["same","valid"]
            context_f1 = [-1*time_stride, 0]
            context_f2 = [0,time_stride]
            if padding == "same" and time_stride % 2 == 1:
                padding_f1 = int(time_stride) // 2 
                padding_f2 = int(time_stride) // 2 + 1
            elif padding == "same" and time_stride % 2 == 0:
                padding_f1 = padding_f2 = int(time_stride) // 2 
            elif padding == "valid":
                padding_f1 = padding_f2 = 0
        
        self.factor1 = SemiOrthogonalConv(input_dim, bottleneck_dim, context_f1, padding_f1)
        self.factor2 = TDNN(bottleneck_dim, output_dim, context_f2, padding_f2)
        self.nl = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.output_dim)
        self.dropout = SharedDimScaleDropout(alpha=0.5, dim=1)

    def forward(self, x, semi_ortho_step=True):
        """
        :param x: is one batch of data, x.size(): [batch_size, input_dim, in_seq_length]
            sequence length is the dimension of the arbitrary length data
        :param semi_ortho_step: if True, update parameter for semi-orthogonality
        :return: [batch_size, output_dim, out_seq_length]
        """
        x = self.factor1(x, semi_ortho_step=semi_ortho_step)
        x = self.factor2(x)
        x = self.nl(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


class TDNNFs(nn.Module):

    def __init__(self, in_dim=30):
        '''
        The FTDNN architecture from
        "State-of-the-art speaker recognition with neural network embeddings in 
        NIST SRE18 and Speakers in the Wild evaluations"
        https://www.sciencedirect.com/science/article/pii/S0885230819302700
        '''
        super(TDNNFs, self).__init__()
        self.layer01 = TDNN(input_dim=in_dim,output_dim=512,context=[-2,-1,0,1,2],padding=2)
        self.layer02 = TDNNF(512,1024,256,2)
        self.layer03 = TDNNF(1024,1024,256,0)
        self.layer04 = TDNNF(1024,1024,256,3)
        self.layer05 = TDNNF(2048,1024,256,0)
        self.layer06 = TDNNF(1024,1024,256,3)

        self.layer07 = TDNNF(3072,1024,256,3)

        self.layer08 = TDNNF(1024,1024,256,3)
 
        self.layer09 = TDNNF(3072,1024,256,0)

        self.layer10 = DenseReLU(1024, 2048)

        self.layer11 = StatsPool()

        self.layer12 = DenseReLU(4096, 512)

    def forward(self, x):
        '''
        Input must be (batch_size,in_dim,seq_len)
        '''
        x = self.layer01(x)
        x_2 = self.layer02(x)
        x_3 = self.layer03(x_2)
        x_4 = self.layer04(x_3)
        skip_5 = torch.cat([x_4, x_3], dim=-2)
        x = self.layer05(skip_5)
        x_6 = self.layer06(x)
        skip_7 = torch.cat([x_6, x_4, x_2], dim=-2)
        x = self.layer07(skip_7)
        x_8 = self.layer08(x)
        skip_9 = torch.cat([x_8, x_6, x_4], dim=-2)
        x = self.layer09(skip_9)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        return x
