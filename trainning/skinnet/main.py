from mobile_former import mobile_former_26m
import torch 

model = mobile_former_26m()

state_dict = torch.load("mobile-former-26m.pth",  map_location=torch.device('cpu'))["state_dict"]

a = model.load_state_dict(state_dict)

print(a)