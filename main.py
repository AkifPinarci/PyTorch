import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# print(torch.__version__)

# Creating Tensors
# Scalar

# scalar = torch.tensor(7)
# print(scalar)
# print(scalar.ndim)
# # Get tensor back as python int
# print(scalar.item())

# Vector
# vector = torch.tensor([7, 7])
# print(vector)
# print(vector.ndim)
# print(vector.shape)

# Matrix
# MATRIX = torch.tensor([[7, 8],
#                         [9, 10]])
# print(MATRIX)
# print(MATRIX.ndim)
# print(MATRIX[1])
# print(MATRIX.shape)

# TENSOR
# TENSOR = torch.tensor([[[1, 2, 3],
#                         [3, 6, 9], 
#                         [2, 4, 5]]])
# print(TENSOR)
# print(TENSOR.ndim)
# print(TENSOR.shape)
# print(TENSOR[0])
# print(TENSOR[0, 1])
# print(TENSOR[0, 1, 1])  

### Random Tensors
# Start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers

#Create a random tensors of size (3, 4)
# random_tensor = torch.rand(3, 4)
# print(random_tensor)
# print(random_tensor.ndim)

# Create a random tensor with similar shape to an image tensor
# random_image_size_tensor = torch.rand(size = (3, 224, 224)) # color channels, height, width
# print(random_image_size_tensor.shape)
# print(random_image_size_tensor.ndim)
# print(random_image_size_tensor)

# Create tensor with all zeros
# zeros = torch.zeros(3, 4)
# print(zeros[0] * random_tensor)

# Create a tensor with all ones
# ones = torch.ones(3, 4)
# print(ones)
# print(ones.dtype)

# Creating a range of tensros and tensors-like
# one_to_ten = torch.arange(start = 1, end = 11, step = 1)

# Creating tensors_like
# ten_zeros = torch.zeros_like(one_to_ten)
# one_zeros = torch.ones_like(one_to_ten)


# Float 32 Tensor
# 1. Tensors not right datatypes
# 2. Tensors not right shape
# 3. Tensors not on the right device 
# float_32_tensor = torch.tensor([3.0, 6.0, 9.0], 
#                                dtype= None, # What data type is the tensor
#                                device = None, # What device is your tensor on
#                                requires_grad=False) # Whether or not to track gradients with this tensors operations
# print(float_32_tensor)

# flaot_16_tensor = float_32_tensor.type(torch.float16)
# print(flaot_16_tensor * float_32_tensor)

# int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.long)
# print(int_32_tensor * float_32_tensor)

# Getting information from tensors
# 1. Tensors not right datatypes - to get datatype from a tensor, can use "tensor.dtype"
# 2. Tensors not right shape - to get shape from a tensor, can use "tensor.shape"
# 3. Tensors not on the right device - to get device from a tensor, can use "tensor.device"

# Create tensors
# tensor = torch.tensor([1, 2, 3])
# print(f"Datatype of tensor: {some_tensor.dtype}")
# print(f"Shape of tensor: {some_tensor.shape}")
# print(f"Device tensor is on: {some_tensor.device}")

# Tensor operations
# Addition - 
# Substraction - 
# Multiplication(element-wise)
# Division
# Matrix multiplicaiton
# print(tensor + 10)
# print(tensor * 10)
# print(tensor - 10)
# print(tensor / 10)

# tensor = torch.mul(tensor, 10)
# print(tensor)

# Matrix multiplication
# Two main ways of performing multiplication in neural newworks and deep learning:
# 1. Element-wise multiplication
# 2. Matrix multiplication (Dot Product)
# tensor = torch.tensor([1, 2, 3])

# print(torch.matmul(tensor, tensor))
# print(tensor @ tensor)

# # .T takes the transpose of the tensor
# print(torch.matmul(torch.rand(3, 2).T, torch.rand(3, 2)))

# Finding the min, max, mean, sum, etc (tensor aggregation)
# x = torch.arange(0, 100, 10)
# print(x.dtype)
# print(x)
# print(torch.min(x))
# print(x.min())
# print(torch.max(x))
# print(x.max())
# print(torch.mean(x.type(torch.float32)))
# print(x.type(torch.float32).mean())
# print(torch.sum(x))
# print(x.sum())

# Finding the positional min and max
# x = torch.arange(0, 100, 10)
# Find the position in tensor that hhas the minimum value with the argmin() -> returns index position of target tensor where the min value occurs
# print(x.argmin())
# print(x.argmax())

# Reshaping, stacking, squezing and unsqueezing tensors
# x = torch.rand(1, 10)
# Reshape - an input tensor to defined shape (reshape has to be compatible with the original shape)
# print(x)
# print(x.shape)
# print(x_reshaped)


# View - Return a view of an input tensor of certain shape but keep the same memory as the original tensor
# z = x.view(1, 9)
# print(z)
# print(z.shape)
# # Changing z changes x (because view of a tensor shares the same memory as the original input) 
# z[:, 0] = 5
# print(x)
# print(z)

# Stacking - combine multiple tensors on top of each other (vstack) or side by side(hstack)
# x_stacked = torch.stack([x, x, x, x])
# print(x_stacked)

# Squeeze - removes all '1' dimension from a tensor
# Unsequeeze - add a  '1' dimension to a target tensor
# x = torch.zeros(2, 1, 2, 1, 2)
# print(x.shape)
# y_squeezed = torch.squeeze(x)
# print(y_squeezed.shape)
# print(y_squeezed)
# y_unsqueezed = y_squeezed.unsqueeze(dim = 1)
# print(y_unsqueezed.shape)
# print(y_unsqueezed)


# Permute - Return a view of the input with dimension permuted (swapped) in a certain way 
# x = torch.randn(2, 3, 5)
# print(x.size())
# y = torch.permute(x, (2, 0, 1))
# print(y.size())
# print(y)

# x_original = torch.rand(size=(224, 224, 3)) # Height, width, color_channels
# x_permuted = x_original.permute((2, 0, 1))
# # print(f"Original: {x_original}")
# # print(f"Permuted: {x_permuted}")
# x_original[0, 0, 1] = 1
# print(x_original[0, 0, 1])
# print(x_permuted[0, 0, 1])
# # print(f"Changed Original: {x_original}")
# # print(f"Changed Permuted: {x_permuted}")

# Indexing(Selecting data from tensors)

