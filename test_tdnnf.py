import torch
from models.tdnnf import TDNNFs

def test_tdnnf():
    tdnn_f = TDNNFs(40)
    # This is a sequence of three 2x1 convolutions
    # dimensions go from 1280 -> 256 -> 256 -> 512
    # dilations and paddings handles how much to dilate and pad each convolution
    # Having these configurable is to ensure the sequence length stays the same
    test_input = torch.rand(5,40,100)
    print(tdnn_f(test_input).shape) # returns (5, 100, 512)

def test_tdnn():
    tdnn = TDNN(40,256,[-2,-1,0,1,2])
    test_input = torch.rand(5,40,100)
    print(tdnn(test_input).shape)

#test_tdnn()
if __name__=="__main__":
    test_tdnnf()