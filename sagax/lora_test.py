# ----- shape sanity test -----
import torch
from torch import nn
from lora import LoRALinear

in_f = 5
out_f = 3
N = 2000

# # Create input
X = torch.rand(N, in_f)

# # Create true weight
W_true = torch.randn(out_f, in_f)

# # Generate target
Y = X @ W_true.T

X_train, X_test = X[:1500], X[1500:]
Y_train, Y_test = Y[:1500], Y[1500:]

base = nn.Linear(in_f, out_f)

# Important: freeze base weights
for p in base.parameters():
    p.requires_grad = False

# wrap with LoRA
model = LoRALinear(base, r=4, alpha=8)

for name, p in model.named_parameters():
    print(name, p.requires_grad, p.shape)

optimizer = torch.optim.AdamW(model.get_lora_parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

for epoch in range(200):
    idx = torch.randint(0, 1500, (64,))   #random minibatch

    x_batch = X_train[idx]
    y_batch = Y_train[idx]

    optimizer.zero_grad()
    pred = model(x_batch)
    loss = loss_fn(pred, y_batch)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.4f}")

test_pred = model(X_test)
test_loss = loss_fn(test_pred, Y_test)
print("Test loss:", test_loss.item())