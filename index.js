import { NeuralNetwork } from "./src/ai.js";

const config = {
    inputSize: 2,
    hiddenLayers: [2],
    outputSize: 1,
    accuracyRate: 0.99,
    log: true,
    activation: "sigmoid",
    save: false,
};

const nn = new NeuralNetwork(config);

nn.train([
    {input: [0,0], output: [0]},
    {input: [0,1], output: [0]},
    {input: [1,0], output: [0]},
    {input: [1,1], output: [1]},
])

let result = nn.run([1, 1])
console.log(result);