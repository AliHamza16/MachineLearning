export class Neuron {
    constructor({ inputs = [], weights = [], bias = 0 }) {
        this.inputs = inputs;
        this.weights = weights;
        this.bias = bias;
    }
    output(activation) {
        const outputValue = this.inputs.reduce((x, y, i) => {
            return x + (y * this.weights[i])
        }, this.bias);
        return activation(outputValue)
    }
    change({ inputs, weights, bias }) {
        this.inputs = inputs || this.inputs;
        this.weights = weights || this.weights;
        this.bias = bias || this.bias;
    }
}