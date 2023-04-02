export const activations = {
    sigmoid: x => 1/(1 + Math.E ** -x),
    tanh: x => Math.tanh(x),
    relu: x => Math.max(0, x),
    softplus: x => Math.log(1 + Math.E ** x),
    gaussian: x => Math.E ** -(x**2),
    binaryStep: x => x < 0 ? 0 : 1
}