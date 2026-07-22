import torch
print(torch.__version__)       # should show something like 2.11.0.dev...+cu128
print(torch.version.cuda)      # should show 12.8
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

x = torch.rand((1000, 1000), device="cuda")
y = torch.mm(x, x)
print(y.device)  # should print cuda:0
import torch, time

x = torch.rand((5000, 5000), device="cuda")
start = time.time()
y = torch.mm(x, x)
torch.cuda.synchronize()
print("GPU time:", time.time() - start)

x_cpu = torch.rand((5000, 5000))
start = time.time()
y_cpu = torch.mm(x_cpu, x_cpu)
print("CPU time:", time.time() - start)
