import { sigmoid } from "./activations.js";

class Neuron {
    constructor({ inputs = [], weights = [], bias = 0 }) {
        this.inputs = inputs;
        this.weights = weights;
        this.bias = bias;
    }
    output(activation = sigmoid) {
        let outputValue = 0;
        for (let index = 0; index < this.inputs.length; index++) {
            const input = this.inputs[index];
            outputValue += input * this.weights[index];
        }
        return activation(outputValue + this.bias);
    }
    change({ inputs, weights, bias }) {
        this.inputs = inputs || this.inputs;
        this.weights = weights || this.weights;
        this.bias = bias || this.bias;
    }
}

export class NeuralNetwork {
    constructor({
        inputSize,
        hiddenLayers,
        outputSize,
        accuracyRate,
        iterations,
    }) {
        this.inputs = Array(inputSize);
        this.hiddenLayer = hiddenLayers.map((size) => {
            return this.#createHiddenLayer(size);
        });
        this.outputLayer = this.#createHiddenLayer(outputSize);
        this.layers = [this.inputs, ...this.hiddenLayer, this.outputLayer];
        this.#randomBrain();
        this.accuracyRate = 1 - accuracyRate;
        this.iterations = iterations;
    }
    #createHiddenLayer = (size) => {
        return Array(size);
    };
    updateLayers() {
        this.layers = [this.inputs, ...this.hiddenLayer, this.outputLayer];
    }
    #createArray(min, max, size) {
        const arr = [];
        for (let index = 0; index < size; index++) {
            arr.push(Math.random() * (max - min) + min);
        }
        return arr;
    }
    getError(actualValues, targetValues) {
        let error = 0;
        for (let index = 0; index < actualValues.length; index++) {
            const element = actualValues[index];
            error += (targetValues[index] - element) ** 2;
        }
        return error;
    }
    #randomBrain() {
        if (this.hiddenLayer[0].length != 0) {
            for (let i = 0; i < this.hiddenLayer.length; i++) {
                for (let j = 0; j < this.hiddenLayer[i].length; j++) {
                    this.hiddenLayer[i][j] = new Neuron({
                        inputs: [],
                        weights: this.#createArray(
                            -15,
                            15,
                            this.layers[i].length
                        ),
                        bias: this.#createArray(-15, 15, 1)[0],
                    });
                }
            }
            for (let i = 0; i < this.outputLayer.length; i++) {
                this.outputLayer[i] = new Neuron({
                    inputs: [],
                    weights: this.#createArray(
                        -10,
                        10,
                        this.layers[this.layers.length - 2].length
                    ),
                    bias: this.#createArray(-15, 15, 1)[0],
                });
            }
        } else {
            for (let i = 0; i < this.outputLayer.length; i++) {
                this.outputLayer[i] = new Neuron({
                    inputs: [],
                    weights: this.#createArray(-15, 15, this.layers[0].length),
                    bias: this.#createArray(-15, 15, 1)[0],
                });
            }
        }
        this.updateLayers();
    }
    run(input) {
        this.inputs = input;
        this.updateLayers();
        let clone = [input];
        if (this.hiddenLayer[0].length != 0) {
            for (let index = 0; index < this.hiddenLayer.length; index++) {
                const a1 = [];
                const element = this.hiddenLayer[index];
                for (let index2 = 0; index2 < element.length; index2++) {
                    const element2 = element[index2];
                    element2.change({ inputs: clone[index] });
                    a1.push(element2.output(sigmoid));
                }
                clone.push(a1);
            }
        }
        const a2 = [];
        for (let index = 0; index < this.outputLayer.length; index++) {
            const element = this.outputLayer[index];
            element.change({ inputs: clone[clone.length - 1] });
            a2.push(element.output(sigmoid));
        }
        clone.push(a2);
        return clone[clone.length - 1];
    }
    train(trainingData) {
        const inputs = trainingData.map((item) => item.input)
        const outputs = trainingData.map((item) => item.output)
        const optimize = () => {
            this.#randomBrain();
            const actualValues = inputs.map((input) => {
                return this.run(input);
            });
            return this.getError(actualValues, outputs);
        };
        let error = 1;
        let t = this.accuracyRate;
        let i = 1;
        while (error > t) {
            error = optimize();
            i++;
            if (!(i % 1000)) {
                t += 0.0001;
                if (this.iterations) console.log(i);
            }
        }
        console.log(
            `eğitim tamamlandı... doğruluk: %${(100 - error * 100).toFixed(2)}`
        );
    }
}
