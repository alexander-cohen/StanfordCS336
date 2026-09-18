import torch
from torch import nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x

def basic_test():
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(s)

    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)

def model_test():
    if not torch.cuda.is_available():
        print("Skipping mixed-precision model test: a CUDA GPU is required.")
        return

    torch.manual_seed(0)
    model = ToyModel(in_features=16, out_features=4).to(
        device="cuda", dtype=torch.float32
    )
    inputs = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    targets = torch.randint(4, (8,), device="cuda")

    def print_output_dtype(name):
        def hook(module, args, output):
            print(f"{name} output: {output.dtype}")
        return hook

    handles = [
        model.fc1.register_forward_hook(print_output_dtype("fc1")),
        model.ln.register_forward_hook(print_output_dtype("layer norm")),
    ]
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for name, parameter in model.named_parameters():
                print(f"Parameter {name} inside autocast: {parameter.dtype}")
            logits = model(inputs)
            print(f"Logits: {logits.dtype}")
            # The handout does not specify a loss; use cross entropy.
            loss = nn.functional.cross_entropy(logits, targets)
            print(f"Cross-entropy loss: {loss.dtype}")

        # This probes gradient dtypes for one backward pass, not a training loop.
        loss.backward()
        for name, parameter in model.named_parameters():
            print(f"Gradient {name}: {parameter.grad.dtype}")
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    print("Accumulation examples:")
    basic_test()
    print("\nFP16 autocast dtype experiment:")
    model_test()
