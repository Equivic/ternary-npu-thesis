from golden_model.ternary_reference import ternary_neuron

import random

inputs = []
weights = []

for _ in range(128):
    inputs.append(random.choice([0, 1]))

for _ in range(32):
    temp_weights = []
    for _ in range(128):
        temp_weights.append(random.choice([-1, 0, 1]))
    weights.append(temp_weights)

print(len(weights))
print(len(weights[0]))
print(len(inputs))

result = []
for x in range(32):
    result.append(ternary_neuron(weights[x], inputs))

print(result)
print(len(result))
