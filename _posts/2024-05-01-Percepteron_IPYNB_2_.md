---
title: Perceptatron Planning
categories: ['Lab Notebook']
tags: ['planning']
description: The planning and information of perceptron
toc: True
comments: True
---

# What is A Perceptron 
- Using a perceptron we are trying to get somewhere by training a model to train by hand signals through the CNN, however it is important that we keep progress and track plans here

# Problems

- Using a perceptron may not be the best becuase it is a very simple training model and through hand signals it may become harder to train.

## Sample of Perceptron

- This Java program implements a perceptron, a basic type of artificial neural network used for binary classification tasks. It trains the perceptron on a small dataset with two input features and corresponding binary labels, adjusting its weights and bias to make predictions and classify new input data based on the learned pattern. Similarly this is how we would use a perceptron to recognize facial signals as a login.


```python
import java.util.Arrays;

public class Perceptron {
    private double[] weights;
    private double bias;
    private double learningRate;

    public Perceptron(int numFeatures, double learningRate) {
        this.weights = new double[numFeatures];
        this.bias = 0;
        this.learningRate = learningRate;
        // Initialize weights with small random values or zeros
        for (int i = 0; i < numFeatures; i++) {
            weights[i] = Math.random() * 0.1;
        }
    }

    public int predict(double[] inputs) {
        double weightedSum = dotProduct(inputs, weights) + bias;
        return weightedSum > 0 ? 1 : 0;  // Step function activation
    }

    public void train(double[][] X_train, int[] y_train, int numEpochs) {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            for (int i = 0; i < X_train.length; i++) {
                double[] inputs = X_train[i];
                int target = y_train[i];
                int prediction = predict(inputs);
                int error = target - prediction;
                // Update weights and bias
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * error * inputs[j];
                }
                bias += learningRate * error;
            }
        }
    }

    private double dotProduct(double[] a, double[] b) {
        double product = 0;
        for (int i = 0; i < a.length; i++) {
            product += a[i] * b[i];
        }
        return product;
    }

    public static void main(String[] args) {
        double[][] X_train = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] y_train = {0, 0, 0, 1};

        Perceptron perceptron = new Perceptron(2, 0.1);
        perceptron.train(X_train, y_train, 100);

        // Test the trained perceptron
        System.out.println("Testing predictions:");
        double[][] X_test = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        for (double[] inputs : X_test) {
            int prediction = perceptron.predict(inputs);
            System.out.println("Inputs: " + Arrays.toString(inputs) + " Predicted: " + prediction);
        }
    }
}


```
