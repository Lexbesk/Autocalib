import torch
from gsplat import rasterization
# Initialize a 3D Gaussian :
mean = torch.tensor([[0. ,0. ,0.01]] , device ="cuda", requires_grad=True)
quat = torch.tensor([[1. ,0. ,0. ,0.]] , device ="cuda", requires_grad=True) # quaternion
color = torch.rand((1 , 3) , device ="cuda")
opac = torch.ones((1 ,) , device ="cuda")
scale = torch.rand((1 , 3) , device ="cuda")
view = torch.eye (4 , device ="cuda") [ None ]
K = torch.tensor([[[1. , 0. , 120.] , [0. , 1. , 120.] , [0. , 0. , 1.]]] , device ="cuda", requires_grad=True) # camera intrinsics

# Render an image using gsplat :

rgb_image , alpha , metadata = rasterization(mean , quat , scale , opac , color , view , K , 240 , 240)

loss = torch.sum(rgb_image)
loss.backward()
print("Gradient of K: ", K.grad)           # None
print("Gradient of mean: ", mean.grad)     # Has gradient
print("Gradient of quat: ", quat.grad)     # Has gradient