import torch 
from mobile_former import mobile_former_26m
import torch.nn as nn
from ghost import GhostModule
from fvcore.nn import FlopCountAnalysis

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

def get_mobile_former(pretrained=False):
    model = mobile_former_26m()
    if pretrained:
        state_dict = torch.load("mobile-former-26m.pth",  map_location=torch.device('cpu'))["state_dict"]
        model.load_state_dict(state_dict)
    return model

def get_skinnet_v1(pretrained=False):
    
    model = mobile_former_26m()
    if pretrained:
        state_dict = torch.load("mobile-former-26m.pth",  map_location=torch.device('cpu'))["state_dict"]
        model.load_state_dict(state_dict)
        
    
    model.features[1].conv1[0] = Identity()
    model.features[1].conv1[1] = Identity()
    model.features[1].se_flag = [0, 0, 2, 0]
    model.features[1].act1 = Identity()
    model.features[1].conv2 = GhostModule(12, 36, relu=False)
    model.features[1].conv3 = GhostModule(36, 12, relu=False)
    
    # print()
    return model
    
    

def get_skinnet_v2(pretrained=False):
    
    model = mobile_former_26m()
    if pretrained:
        state_dict = torch.load("mobile-former-26m.pth",  map_location=torch.device('cpu'))["state_dict"]
        model.load_state_dict(state_dict)
        
    
    model.features[1].conv1[0] = Identity()
    model.features[1].conv1[1] = Identity()
    model.features[1].se_flag = [0, 0, 2, 0]
    model.features[1].act1 = Identity()
    model.features[1].conv2 = GhostModule(12, 36, relu=False)
    model.features[1].conv3 = GhostModule(36, 12, relu=False)
    
    model.features[3].conv1[0] = Identity()
    model.features[3].conv1[1] = Identity()
    model.features[3].se_flag = [0, 0, 2, 0]
    model.features[3].act1 = Identity()
    model.features[3].conv2 = GhostModule(24, 72, relu=False)
    model.features[3].conv3 = GhostModule(72, 24, relu=False)
    
    return model

    
def get_skinnet_v3(pretrained=False):
    
    model = mobile_former_26m()
    if pretrained:
        state_dict = torch.load("mobile-former-26m.pth",  map_location=torch.device('cpu'))["state_dict"]
        model.load_state_dict(state_dict)
        
    
    model.features[1].conv1[0] = Identity()
    model.features[1].conv1[1] = Identity()
    model.features[1].se_flag = [0, 0, 2, 0]
    model.features[1].act1 = Identity()
    model.features[1].conv2 = GhostModule(12, 36, relu=False)
    model.features[1].conv3 = GhostModule(36, 12, relu=False)
    
    model.features[3].conv1[0] = Identity()
    model.features[3].conv1[1] = Identity()
    model.features[3].se_flag = [0, 0, 2, 0]
    model.features[3].act1 = Identity()
    model.features[3].conv2 = GhostModule(24, 72, relu=False)
    model.features[3].conv3 = GhostModule(72, 24, relu=False)
    
    model.features[5].conv1[0] = Identity()
    model.features[5].conv1[1] = Identity()
    model.features[5].se_flag = [0, 0, 2, 0]
    model.features[5].act1 = Identity()
    model.features[5].conv2 = GhostModule(48, 192, relu=False)
    model.features[5].conv3 = GhostModule(192, 48, relu=False)
    
    model.features[6].conv1[0] = Identity()
    model.features[6].conv1[1] = Identity()
    model.features[6].se_flag = [0, 0, 2, 0]
    model.features[6].act1 = Identity()
    model.features[6].conv2 = GhostModule(48, 288, relu=False)
    model.features[6].conv3 = GhostModule(288, 64, relu=False)
    
    model.features[8].conv1[0] = Identity()
    model.features[8].conv1[1] = Identity()
    model.features[8].se_flag = [0, 0, 2, 0]
    model.features[8].act1 = Identity()
    model.features[8].conv2 = GhostModule(96, 576, relu=False)
    model.features[8].conv3 = GhostModule(576, 96, relu=False)
    
    return model

def test():
    modelget = get_skinnet_v2()
        
    # print(modelget(torch.ones(2,3,224,224)).shape)
        
    def count_parameters(model):
            total_trainable_params = 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                params = parameter.numel()
                total_trainable_params += params
            return total_trainable_params

    input_shape = (2, 3, 224, 224)

    total_params = count_parameters(modelget)

    print(modelget(torch.ones((input_shape))).shape)

    flops = FlopCountAnalysis(modelget, torch.ones((input_shape), dtype=torch.float32))
    model_flops = flops.total()
    print(f"Total Trainable Params: {round(total_params * 1e-6, 2)} M")
    print(f"MAdds: {round(model_flops * 1e-6, 2)} M")