import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

# parser = argparse.ArgumentParser(description='PyTorch Example')
# parser.add_argument('--disable-cuda', action='store_true',
#                     help='Disable CUDA')
# args = parser.parse_args()
# args.device = None
# if not args.disable_cuda and torch.cuda.is_available():
#     args.device = torch.device('cuda')
# else:
#     args.device = torch.device('cpu')

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

# x = torch.arange(1, 10).reshape(1, 3, 3)
# print(x)

# # print(x[0][2][1])
# # print(x[0, 2, 1])

# # Get all values of 0th and 1st dimensions but only index 1 of 2nd dimension
# print(x[:, :, 1])
# # Get all values of the 0 dim but only the 1 index value of 1st and 2nd dim
# # print(x[:, 1, 1])
# print(x[:, -1, 2])

# PyTorch tensors & NumPy
# Data in NumPy, want in PyTorch tensor -> "torch.from_numpy(ndarray)"
# PyTorch -> NumPy "torch.Tensor.numpy()"

# NumPy > Torch
# array = np.arange(1.0, 8.0)
# tensor = torch.from_numpy(array).type(torch.float32) # When converting from numpy to pytorch, pytorch reflects numpy's default datatype of float64

# array += 1
# print(tensor)
# print(array) 

# Tensor -> NumPy
# tensor = torch.ones(1, 9)
# numpy_tensor = tensor.numpy()

# print(tensor)
# print(numpy_tensor)

# tensor =tensor + 1
# print(tensor)
# print(numpy_tensor)

# Reproducibility (trying to take random out of random)
# start with random numbers -> tensor operations -> update random numbers to try and make them better representation of the data -> again -> again

# ToReduce randomness in neural networks and PyTorch comes the concept of a "random seed".
# Essentially what the random seed does is "flavour" the randomness.
# random_tensor_A = torch.rand(3, 4)
# random_tensor_B = torch.rand(3, 4)
# print(random_tensor_A)
# print(random_tensor_B)

# Make some random but reproducible tensors
# https://en.wikipedia.org/wiki/Random_seed
# https://pytorch.org/docs/stable/notes/randomness.html#python
# Set the random seed
# RANDOM_SEED = 42
# torch.manual_seed(RANDOM_SEED)
# random_tensor_C = torch.rand(3, 4)

# torch.manual_seed(RANDOM_SEED)
# random_tensor_D = torch.rand(3, 4)

# print(random_tensor_C)
# print(random_tensor_D)
# print(random_tensor_D == random_tensor_C)

# Running tensors and PyTorch objects on the GPUs(and making aster computations)
# Getting GPU
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(device)

# # Moving tensors to the GPU
# tensor = torch.tensor([1, 2, 3])
# tensor_on_gpu = tensor.to(device)
# print(tensor_on_gpu, tensor_on_gpu.device)

# # Moving back tensors to CPU because if a tensor is on a GPU, we cannot transform it to NumPy
# tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
# print(tensor_back_on_cpu)

# 2. Create a random tensor with shape (7, 7).

# # Import torch
# import torch 

# # Create random tensor
# X = torch.rand(size=(7, 7))
# X, X.shape
     
# 3. Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7) (hint: you may have to transpose the second tensor).

# # Create another random tensor
# Y = torch.rand(size=(1, 7))
# # Z = torch.matmul(X, Y) # will error because of shape issues
# Z = torch.matmul(X, Y.T) # no error because of transpose
# Z, Z.shape
     

# 4. Set the random seed to 0 and do 2 & 3 over again.

# # Set manual seed
# torch.manual_seed(0)

# # Create two random tensors
# X = torch.rand(size=(7, 7))
# Y = torch.rand(size=(1, 7))

# # Matrix multiply tensors
# Z = torch.matmul(X, Y.T)
# Z, Z.shape

# 5. Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent? (hint: you'll need to look into the documentation for torch.cuda for this one)
# If there is, set the GPU random seed to 1234.

# # Set random seed on the GPU
# torch.cuda.manual_seed(1234)
     
# 6. Create two random tensors of shape (2, 3) and send them both to the GPU (you'll need access to a GPU for this). Set torch.manual_seed(1234) when creating the tensors (this doesn't have to be the GPU random seed). The output should be something like:


# # Set random seed
# torch.manual_seed(1234)

# # Check for access to GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Device: {device}")

# # Create two random tensors on GPU
# tensor_A = torch.rand(size=(2,3)).to(device)
# tensor_B = torch.rand(size=(2,3)).to(device)
# tensor_A, tensor_B

# 7. Perform a matrix multiplication on the tensors you created in 6 (again, you may have to adjust the shapes of one of the tensors).
# The output should look like:

# # Perform matmul on tensor_A and tensor_B
# # tensor_C = torch.matmul(tensor_A, tensor_B) # won't work because of shape error
# tensor_C = torch.matmul(tensor_A, tensor_B.T)
# tensor_C, tensor_C.shape

# 8. Find the maximum and minimum values of the output of 7.

# # Find max
# max = torch.max(tensor_C)

# # Find min
# min = torch.min(tensor_C)
# max, min

# 9. Find the maximum and minimum index values of the output of 7.

# # Find arg max
# arg_max = torch.argmax(tensor_C)

# # Find arg min
# arg_min = torch.argmin(tensor_C)
# arg_max, arg_min

# 10. Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with all the 1 dimensions removed to be left with a tensor of shape (10). Set the seed to 7 when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape.
# The output should look like:

# # Set seed
# torch.manual_seed(7)

# # Create random tensor
# tensor_D = torch.rand(size=(1, 1, 1, 10))

# # Remove single dimensions
# tensor_E = tensor_D.squeeze()

# # Print out tensors
# print(tensor_D, tensor_D.shape)
# print(tensor_E, tensor_E.shape)