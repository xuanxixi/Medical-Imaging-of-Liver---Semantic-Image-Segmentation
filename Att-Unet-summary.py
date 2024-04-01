#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from torchsummary import summary

#from nets.unet import Unet
from nets.attention_unet import AttU_Net
if __name__ == "__main__":
    model = AttU_Net().train().cuda()
    summary(model,(3,512,512))

