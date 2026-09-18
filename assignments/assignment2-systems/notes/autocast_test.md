Accumulation examples:
tensor(10.0001)
tensor(9.9531, dtype=torch.float16)
tensor(10.0021)
tensor(10.0021)

FP16 autocast dtype experiment:
Parameter fc1.weight inside autocast: torch.float32
Parameter ln.weight inside autocast: torch.float32
Parameter ln.bias inside autocast: torch.float32
Parameter fc2.weight inside autocast: torch.float32
fc1 output: torch.float16
layer norm output: torch.float32
Logits: torch.float16
Cross-entropy loss: torch.float32
Gradient fc1.weight: torch.float32
Gradient ln.weight: torch.float32
Gradient ln.bias: torch.float32
Gradient fc2.weight: torch.float32
