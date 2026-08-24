def ternary_contribution(weight, input_bit):
    if weight == 0:
        contribution = 0
    elif weight > 0:
        contribution = 1 if input_bit == 1 else -1
    else:
        contribution = 1 if input_bit == 0 else -1
    return contribution


def ternary_neuron(weights, inputs):
    total = 0
    for w, i in zip(weights, inputs):
        total += ternary_contribution(w, i)
    return 1 if total > 0 else 0
