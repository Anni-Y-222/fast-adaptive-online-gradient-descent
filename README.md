# fast-adaptive-online-gradient-descent
paper:'A Fast Adaptive Online Gradient Descent Algorithm in Over-Parameterized Neural Networks'
Usage：Replace original algorithm with FAOGD, and your experiment results should improve!

from ours import FAOGD
optimizer = FAOGD(model.parameters(), max_lr=0.01)
optimizer.zero_grad()
loss = torch.nn.functional.cross_entropy(model(x), y)
loss.backward()
optimizer.step(lambda: float(loss))