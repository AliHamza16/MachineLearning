import { activations } from "./activations.js";
import { Neuron } from "./neuron.js";
import fs from "fs";
import {randomUUID} from "crypto"

const range = 15

export class NeuralNetwork {
    constructor({
        inputSize,
        hiddenLayers = [0],
        outputSize,
        accuracyRate = 0.95,
        log = false,
        activation = "sigmoid",
        save = false,
    }) {
        this.inputs = Array(inputSize);
        this.hiddenLayer = hiddenLayers.map((size) => Array(size));
        this.outputLayer = Array(outputSize);
        this.layers = [this.inputs, ...this.hiddenLayer, this.outputLayer];
        this.accuracyRate = 1 - accuracyRate;
        this.log = log;
        this.activation = activations[activation];
        this.save = save;
        this.#randomBrain();
    }
    #updateLayers() {
        this.layers = [this.inputs, ...this.hiddenLayer, this.outputLayer];
    }
    #createArray(min, max, size) {
        return Array(size)
            .fill(null)
            .map((item) => {
                return Math.random() * (max - min) + min;
            });
    }
    #getError(actualValues, targetValues) {
        let error = 0;
        for (let index = 0; index < actualValues.length; index++) {
            const element = actualValues[index];
            const element2 = targetValues[index];
            element.map((x, i) => {
                error += (element2[i] - x) ** 2;
            });
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
                            -range,
                            range,
                            this.layers[i].length
                        ),
                        bias: this.#createArray(-range, range, 1)[0],
                    });
                }
            }
            for (let i = 0; i < this.outputLayer.length; i++) {
                this.outputLayer[i] = new Neuron({
                    inputs: [],
                    weights: this.#createArray(
                        -range,
                        range,
                        this.layers[this.layers.length - 2].length
                    ),
                    bias: this.#createArray(-range, range, 1)[0],
                });
            }
        } else {
            for (let i = 0; i < this.outputLayer.length; i++) {
                this.outputLayer[i] = new Neuron({
                    inputs: [],
                    weights: this.#createArray(-range, range, this.layers[0].length),
                    bias: this.#createArray(-range, range, 1)[0],
                });
            }
        }
        this.#updateLayers();
    }
    run(input) {
        this.inputs = input;
        this.#updateLayers();
        let clone = [input];
        if (this.hiddenLayer[0].length != 0) {
            for (let index = 0; index < this.hiddenLayer.length; index++) {
                const a1 = [];
                const element = this.hiddenLayer[index];
                for (let index2 = 0; index2 < element.length; index2++) {
                    const element2 = element[index2];
                    element2.change({ inputs: clone[index] });
                    a1.push(Number(element2.output(this.activation).toFixed(8)));
                }
                clone.push(a1);
            }
        }
        const a2 = [];
        for (let index = 0; index < this.outputLayer.length; index++) {
            const element = this.outputLayer[index];
            element.change({ inputs: clone[clone.length - 1] });
            a2.push(Number(element.output(this.activation).toFixed(8)));
        }
        clone.push(a2);
        return clone[clone.length - 1];
    }
    train(trainingData) {
        const inputs = trainingData.map((item) => item.input);
        const outputs = trainingData.map((item) => item.output);
        const optimize = () => {
            this.#randomBrain();
            const actualValues = inputs.map((input) => {
                return this.run(input);
            });
            return this.#getError(actualValues, outputs);
        };
        let error = 1;
        let min_error = 1;
        let t = this.accuracyRate;
        let i = 1;
        while (error > t) {
            error = optimize();
            if (error < min_error) {
                min_error = error;
                if (this.log) console.log(i, `error: ${min_error}`);
            }
            i++;
            if (!(i % 1000)) {
                t += 0.0001;
                if(this.log) console.log(i);
            }
        }
        console.log(
            `training completed... accuracy: %${(100 - error * 100).toFixed(2)}`
        );
        if (this.save) {
            this.#updateLayers();
            const a = randomUUID()
            fs.writeFile(
                `models/${a}.json`,
                JSON.stringify(this.layers),
                (err) => {
                    if (err) {
                        console.log(err);
                    } else {
                        console.log("saved", `${a}.json`);
                    }
                }
            );
        }
    }
}
