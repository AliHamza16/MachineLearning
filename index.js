import { NeuralNetwork } from "./src/ai.js";

const config = {
    inputSize: 2,
    hiddenLayers: [0],
    outputSize: 1,
    accuracyRate: 0.999,
    iterations: false
}

let nn = new NeuralNetwork(config)

let trainingData = [
    {input: [0,0], output: 1},
    {input: [0,1], output: 1},
    {input: [1,0], output: 0},
    {input: [1,1], output: 1},
]
nn.train(trainingData) // p => q
console.log(nn.run([1, 0])); // 0.006203247493668751