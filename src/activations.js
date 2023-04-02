export const sigmoid = x => 1/(1 + Math.E ** -x)
export const binaryStep = x => x < 0 ? 0 : 1
export const tanh = x => Math.tanh(x)
export const relu = x => Math.max(0, x)
export const softplus = x => Math.log(1 + Math.E ** x)
export const gaussian = x => Math.E ** -(x**2)