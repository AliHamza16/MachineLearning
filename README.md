# MachineLearning
Create and train neural networks.

> **Note:** For now, the brain trains itself randomly, so if you use networks that are too complex, the training time may increase.

# Neural Network Configurations
## inputSize
Number of input neurons in your neural network.
## outputSize
Number of output neurons in your neural network.
## hiddenLayers
Hidden units in your neural network.
<br>
```javascript
const config = {
    ...,
    inputSize: 2,
    hiddenLayers: [4, 4],
    outputSize: 1
}
```
> Each element of the array specifies how many neurons will be in that layer
<img src="https://upload.wikimedia.org/wikipedia/commons/d/d2/Neural_network_explain.png" />

## accuracyRate
The closer the ratio is to one, the higher the accuracy of the neural network.
However, high accuracy rates can lead to longer training times for complex networks.

## iterations
Determines whether to print to the console how many iterations have been done.

# Training Neural Network
```javascript
const config = {
    inputSize: 2,
    hiddenLayers: [0],
    outputSize: 1,
    accuracyRate: 0.999,
    iterations: false
}

const nn = new NeuralNetwork(config)
```
to train the neural network, create a array that specifies which result it will give based on which input
```javascript
let trainingData = [
    {input: [0,0], output: 1},
    {input: [0,1], output: 1},
    {input: [1,0], output: 0},
    {input: [1,1], output: 1},
]
```
and call train function with training data
```javascript
nn.train(trainingData)
```
then call run function to get result
```javascript
let result = nn.run([1, 1]) 
// [ 0.9876001252 ]
```
the run function returns the output layer of your neural network for given inputs
<br>
>in this example we trained the neural network to learn the conditional statement **"if p then q"**
