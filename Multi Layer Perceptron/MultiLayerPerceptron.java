import java.io.File;
import java.io.FileNotFoundException;
import java.util.Random;
import java.util.Scanner;

public class MultiLayerPerceptron {
    // This class implements a MultiLayerPerceptron model to classify images of handwritten digits
    // It has 1 input layer, 1 hidden layer and 1 output layer with 64, 512 and 10 perceptrons respectively
    // It has a constructor to initialize the weights and biases for the model
    // It initializes the weights and biases using random numbers from a Gaussian distribution

    // It has methods to train the model and test the model
    // It has methods to calculate the accuracy, confusion matrix and misclassification rate
    // It has methods to read the data from a csv file and count the number of lines in a file

    // There are 2 datasets for the model. 
    // We do a 2 fold cross validation for the model using the 2 datasets as training and test data for each fold respectively.
    // This program prints the confusion matrix and the misclassified rate for each digit for each fold
    // It prints the average accuracy for the model on the two folds.

    // It uses the tanh activation function for the hidden layer to get better accuracy
    // It uses the backpropagation algorithm to update the weights and biases after each epoch to minimize the error
    // It uses the cosine decay learning rate schedule to update the learning rate after each epoch to prevent overfitting
    // It uses the dropout technique to prevent overfitting by randomly setting the activations of some hidden perceptrons to 0
    // It uses the softmax function to calculate the probabilities for the output layer

    // The following hyperparameters are use for the model:
    private static final String PATH_TO_CSV_DATASET1 = "cw2DataSet1.csv";//Path to the csv file for the first dataset
    private static final String PATH_TO_CSV_DATASET2 = "cw2DataSet2.csv";//Path to the csv file for the second dataset
    private static final int PIXELS_IN_ROW = 8;//8 pixels in a row
    private static final int PIXEL_COUNT = PIXELS_IN_ROW*PIXELS_IN_ROW;//8x8 pixels in the image = 64 pixels
    private static final int ROW_LENGTH = PIXEL_COUNT+1;//64 pixels + 1 label in the csv file for each row = 65 columns
	private static final double PIXEL_VALUE = 16.0;//16 different pixel values in the image
    private static final int EPOCHS = 145;//Number of epochs for training the model
	private static final int NUMBER_OF_CLASSES = 10;//Number of classes in the dataset
    private static final double ROUNDING_4_PLACES = 10000.0;//This is used to round the number to 4 decimal places
    private static final double CENTURY_FOR_PERCENTAGE = 100.0;//This is used to calculate the percentage
    private static final int INPUT_SIZE = PIXEL_COUNT;// Number of input neurons/perceptrons
    private static final int OUTPUT_SIZE = NUMBER_OF_CLASSES;// Number of output neurons/perceptrons
    private static final int HIDDEN_SIZE = 512 ;// Number of hidden neurons/perceptrons
    private static final double REQUIRED_ACCURACY = 98.5;// Required accuracy for the model
    private static final double MAX_LEARNING_RATE = .2;// Learning rate for the model
    private static final double MIN_LEARNING_RATE = 0;// Learning rate for the model
    private static final long SEED = 0l;//Seed for the random number generator to generate the same random numbers each time the program is run
    private static final double STANDARD_DEVIATION = 0.1;//Standard deviation for the random numbers from a Gaussian distribution
    private static final double MEAN = 0;//Mean for the random numbers from a Gaussian distribution
    private static final double TWO_FOR_AVG = 2.0;//This is used to calculate the average accuracy
    private static final double POWER_OF_2 = 2.0;//This is used to calculate the derivative of the tanh function
    private static final double DROPOUT_RATE = 0.51; // Dropout rate for the hidden layer
    private static final double HALF = 0.5;//This is used to calculate the cosine decay
    private static final double STOP_THRESHOLD = 10;//This is used to stop the training if the accuracy drops by more than 10% from the maximum accuracy
    private static final int SIGMOID = 1;//This is used to calculate the derivative of the sigmoid function
    private static final int TANH = 2;//This is used to calculate the derivative of the tanh function
    private static final int DERIVATIVE = TANH;//This is used to choose the activation function for the hidden layer

    private double learningRate = MAX_LEARNING_RATE;// Learning rate for the model
    private double[][] inputToHiddenWeights; // Weights between input and hidden layer
    private double[][] hiddenToOutputWeights; // Weights between hidden and output layer
    private double[] hiddenBias;// Biases for the hidden layer. Each hidden perceptron has a bias. The bias is added to the weighted sum of the inputs
    private double[] outputBias;// Biases for the output layer. Each output perceptron has a bias. The bias is added to the weighted sum of the inputs
    private double[] hiddenActivation;// Activation for the hidden layer. The activation is the output of the hidden perceptron after passing the weighted sum of the inputs through the activation function
    private Random random;
    
    // Constructor
    public MultiLayerPerceptron() {
        // This is the constructor for the MultiLayerPerceptron class
        // It initializes the weights and biases for the model

        random = new Random(SEED);//This is used to generate the same random numbers each time the program is run
    
        initializeWeights();
        initializeBiases();
        this.hiddenActivation = new double[HIDDEN_SIZE];//Activation for the hidden layer
    }

    private void initializeWeights() {
        // This function initializes the weights for the input to hidden layer and the hidden to output layer
        // The weights are initialized using random numbers from a Gaussian distribution with a mean of 0 and a standard deviation of 0.1
        
        this.inputToHiddenWeights = new double[INPUT_SIZE][HIDDEN_SIZE];//Weights between input and hidden layer
        this.hiddenToOutputWeights = new double[HIDDEN_SIZE][OUTPUT_SIZE];//Weights between hidden and output layer
        
        for (int inputPerceptronIndex = 0; inputPerceptronIndex < INPUT_SIZE; inputPerceptronIndex++) {//For each input perceptron
            for (int hiddenPerceptronIndex = 0; hiddenPerceptronIndex < HIDDEN_SIZE; hiddenPerceptronIndex++) {//For each hidden perceptron
                this.inputToHiddenWeights[inputPerceptronIndex][hiddenPerceptronIndex] = random.nextGaussian() * STANDARD_DEVIATION + MEAN;//Initialize the weights using random numbers from a Gaussian distribution with a mean and a standard deviation
            }
        }
        
        for (int hiddenPerceptronIndex = 0; hiddenPerceptronIndex < HIDDEN_SIZE; hiddenPerceptronIndex++) {//For each hidden perceptron
            for (int outputPerceptronIndex = 0; outputPerceptronIndex < OUTPUT_SIZE; outputPerceptronIndex++) {//For each output perceptron
                this.hiddenToOutputWeights[hiddenPerceptronIndex][outputPerceptronIndex] = random.nextGaussian() * STANDARD_DEVIATION+ MEAN;//Initialize the weights using random numbers from a Gaussian distribution with a mean and a standard deviation
            }
        }
    }
    
    private void initializeBiases() {
        // This function initializes the biases for the hidden and output layers
        // The biases are initialized using random numbers from a Gaussian distribution with a mean of 0 and a standard deviation of 0.1

        this.hiddenBias = new double[HIDDEN_SIZE]; // Biases for the hidden layer
        this.outputBias = new double[OUTPUT_SIZE]; // Biases for the output layer

        for (int hiddenPerceptronIndex = 0; hiddenPerceptronIndex < HIDDEN_SIZE; hiddenPerceptronIndex++) { //For each hidden perceptron
            this.hiddenBias[hiddenPerceptronIndex] = random.nextGaussian() * STANDARD_DEVIATION + MEAN;//Initialize the bias using random numbers from a Gaussian distribution with a mean and a standard deviation
        }

        for (int outputPerceptronIndex = 0; outputPerceptronIndex < OUTPUT_SIZE; outputPerceptronIndex++) { //For each output perceptron
            this.outputBias[outputPerceptronIndex] = random.nextGaussian() * STANDARD_DEVIATION + MEAN;//Initialize the bias using random numbers from a Gaussian distribution with a mean and a standard deviation
        }
    }
    
    public void cosineDecay(int currentEpoch) {
        // This function updates the learning rate using the cosine decay schedule
        // The learning rate is updated after each epoch
        // The learning rate is updated using the formula: min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * current_epoch / epochs))
        // The learning rate is updated to prevent overfitting and to improve the accuracy of the model

        learningRate = MIN_LEARNING_RATE + (HALF) * (MAX_LEARNING_RATE - MIN_LEARNING_RATE) * (1 + Math.cos(Math.PI * currentEpoch / EPOCHS));
    }

    public void train(double[][] inputs, int[] targets, int epochs) {
        // This function trains the model using the inputs and targets
        // It caches the weights and biases for the model with the highest accuracy
        // It updates the weights and biases using backpropagation after each epoch
        // It updates the learning rate using the cosine decay schedule after each epoch

        int epoch = 0;//Current epoch
        double cachedInputToHiddenWeights[][] = new double[INPUT_SIZE][HIDDEN_SIZE];
        double cachedHiddenToOutputWeights[][] = new double[HIDDEN_SIZE][OUTPUT_SIZE];
        double cachedHiddenBias[] = new double[HIDDEN_SIZE];
        double cachedOutputBias[] = new double[OUTPUT_SIZE];
        double cachedHiddenActivation[] = new double[HIDDEN_SIZE];
        double maxAccuracy = 0;

        while(epoch<epochs){
            cosineDecay(epoch);//Update the learning rate using the cosine decay schedule
            epoch++;//Increment the epoch
            double accuracy = trainAndGetAccuracy(inputs, targets, epochs, maxAccuracy);
            if(accuracy>maxAccuracy){
                maxAccuracy = accuracy;
                copyWeights(cachedInputToHiddenWeights, inputToHiddenWeights);
                copyWeights(cachedHiddenToOutputWeights, hiddenToOutputWeights);
                copyBiases(cachedHiddenBias, hiddenBias);
                copyBiases(cachedOutputBias, outputBias);
                copyActivation(cachedHiddenActivation, hiddenActivation);
            }
            System.out.println("Epoch: "+epoch +" Accuracy: "+accuracy);//Print the epoch
            if(stopEarly(accuracy, epoch, maxAccuracy)){
                break;
            }
        }

        //Restore the weights and biases for the model with the highest accuracy
        copyWeights(inputToHiddenWeights, cachedInputToHiddenWeights);
        copyWeights(hiddenToOutputWeights, cachedHiddenToOutputWeights);
        copyBiases(hiddenBias, cachedHiddenBias);
        copyBiases(outputBias, cachedOutputBias);
        copyActivation(hiddenActivation, cachedHiddenActivation);
    }

    double trainAndGetAccuracy(double[][] inputs, int[] targets, int epochs, double maxAccuracy){
        // This function trains the model using the inputs and targets and uses backpropagation to update the weights and biases
        // It returns the accuracy of the model on the inputs

        for (int inputIndex = 0; inputIndex < inputs.length; inputIndex++) {
            double[] input = inputs[inputIndex];//Get the input
            int target = targets[inputIndex];//Get the target
            double[] output = forwardPropagation(input,true);//Forward propagation to get the output
            double[] outputError = new double[OUTPUT_SIZE];//Error in the output layer
            outputError[target] = 1 - output[target];//Error in the output layer for the target class is 1 - output for the target class
            backPropagation(outputError, input);//Backpropagation to update the weights and biases using the error in the output layer
        }

        int correct = 0;//Number of correct predictions
        for (int inputIndex = 0; inputIndex < inputs.length; inputIndex++) {
            double[] input = inputs[inputIndex];//Get the input
            int target = targets[inputIndex];//Get the target
            int prediction = predict(input);//Get the prediction
            if (prediction == target) {//If the prediction is correct
                correct++;//Increment the number of correct predictions
            }
        }
        return correct*CENTURY_FOR_PERCENTAGE/inputs.length;//Get the accuracy
    }

    void copyWeights(double[][] toWeights, double[][] fromWeights){
        // This function copies the weights from one 2D array to another 2D array

        for (int rowIterator = 0; rowIterator < toWeights.length; rowIterator++) {
            for (int colIterator = 0; colIterator < toWeights[0].length; colIterator++) {
                toWeights[rowIterator][colIterator] = fromWeights[rowIterator][colIterator];
            }
        }
    }

    void copyBiases(double[] toBiases, double[] fromBiases){
        // This function copies the biases from one 1D array to another 1D array

        for (int index = 0; index < toBiases.length; index++) {
            toBiases[index] = fromBiases[index];
        }
    }

    void copyActivation(double[] toActivation, double[] fromActivation){
        // This function copies the activations from one 1D array to another 1D array

        for (int index = 0; index < toActivation.length; index++) {
            toActivation[index] = fromActivation[index];
        }
    }
    
    boolean stopEarly(double accuracy, int epoch, double maxAccuracy){
        // This function stops the training if the accuracy is more than the required accuracy for the model 
        // or if the accuracy drops by more than 10% from the maximum accuracy
        // It returns true if the training should stop and false otherwise

        if(accuracy > REQUIRED_ACCURACY){//If the accuracy is more than the required accuracy for the model
            System.out.println("Desired output reached at epoch: "+epoch);//Print the epoch
            System.out.println("Accuracy: "+accuracy+"%");//Print the accuracy
            return true;
        }
        if(maxAccuracy-accuracy>STOP_THRESHOLD){
            return true;
        }
        return false;
    }

    private double[] forwardPropagation(double[] input, boolean isTraining) {
        // This function calculates the activations in the hidden layer and the output layer
        // The activations in the hidden layer are calculated using the input and the weights between the input and hidden layer
        // The activations in the output layer are calculated using the activations in the hidden layer and the weights between the hidden and output layer
        // The biases are added to the activations
        // The activations are passed through the activation function to get the output
        
        double[] dropoutMask = new double[HIDDEN_SIZE];
        double[] hiddenActivationCopy = new double[HIDDEN_SIZE];
        for (int hiddenPerceptronIndex = 0; hiddenPerceptronIndex < HIDDEN_SIZE; hiddenPerceptronIndex++) {
            hiddenActivation[hiddenPerceptronIndex] = 0;
            for (int inputPerceptronIndex = 0; inputPerceptronIndex < INPUT_SIZE; inputPerceptronIndex++) {
                hiddenActivation[hiddenPerceptronIndex] += input[inputPerceptronIndex] * inputToHiddenWeights[inputPerceptronIndex][hiddenPerceptronIndex];
            }
            hiddenActivation[hiddenPerceptronIndex] += hiddenBias[hiddenPerceptronIndex];
            hiddenActivation[hiddenPerceptronIndex] += relu(hiddenActivation[hiddenPerceptronIndex]);
            // Apply dropout if training
            if (isTraining) {
                if (random.nextDouble() < DROPOUT_RATE) {
                    dropoutMask[hiddenPerceptronIndex] = 0;//Set the dropout mask to 0 to drop the activation
                } else {
                    dropoutMask[hiddenPerceptronIndex] = 1;//Set the dropout mask to 1 to keep the activation
                }
                // Apply dropout mask
                hiddenActivationCopy[hiddenPerceptronIndex] = hiddenActivation[hiddenPerceptronIndex];//Copy the activation
                hiddenActivation[hiddenPerceptronIndex] *= dropoutMask[hiddenPerceptronIndex];//Drops the activation if the dropout mask is 0
                dropoutMask[hiddenPerceptronIndex] = 1;//Reset the dropout mask to 1
            }
        }

        // Prepare for Softmax calculation in the output layer
        double[] rawOutput = new double[OUTPUT_SIZE];
        for (int outputPerceptronIndex = 0; outputPerceptronIndex < OUTPUT_SIZE; outputPerceptronIndex++) {
            rawOutput[outputPerceptronIndex] = 0;
            for (int hiddenPerceptronIndex = 0; hiddenPerceptronIndex < HIDDEN_SIZE; hiddenPerceptronIndex++) {
                rawOutput[outputPerceptronIndex] += hiddenActivation[hiddenPerceptronIndex] * hiddenToOutputWeights[hiddenPerceptronIndex][outputPerceptronIndex];
            }
            rawOutput[outputPerceptronIndex] += outputBias[outputPerceptronIndex];
        }
        copyActivation(hiddenActivation, hiddenActivationCopy);
        // Apply Softmax on the raw output to get probabilities
        return softmax(rawOutput);
    }

    private double[] softmax(double[] rawOutput) {
        // This function applies the softmax function to the raw output to get the probabilities
        // Softmax function is the exponential of the raw output divided by the sum of the exponentials of the raw output
        // The probabilities are returned

        double[] softmax = new double[rawOutput.length];//Probabilities
        double sum = 0;//Sum of the exponentials of the raw output
    
        for (int outputIndex = 0; outputIndex < rawOutput.length; outputIndex++) {
            sum += Math.exp(rawOutput[outputIndex]);//Sum of the exponentials of the raw output
        }
        for (int outputIndex = 0; outputIndex < rawOutput.length; outputIndex++) {
            softmax[outputIndex] = Math.exp(rawOutput[outputIndex]) / sum;//Probabilities = exponential of the raw output / sum of the exponentials of the raw output
        }
        return softmax;//Return the probabilities
    }
        

    private void backPropagation(double[] outputError, double[] input) {
        // This function updates the weights and biases using the error in the output layer
        // It calculates the error gradient for the hidden layer
        // It updates the weights for the input to hidden layer
        // It updates the biases for the hidden and output layers

        // Calculate error gradient for hidden layer
        double[] hiddenError = new double[HIDDEN_SIZE];
        for (int hiddenPerceptronIndex = 0; hiddenPerceptronIndex < HIDDEN_SIZE; hiddenPerceptronIndex++) {
            double error = 0;
            for (int outputPerceptronIndex = 0; outputPerceptronIndex < OUTPUT_SIZE; outputPerceptronIndex++) {
                error += outputError[outputPerceptronIndex] * hiddenToOutputWeights[hiddenPerceptronIndex][outputPerceptronIndex];
            }
            if(DERIVATIVE == SIGMOID)
                hiddenError[hiddenPerceptronIndex] = error * sigmoidDerivative(hiddenActivation[hiddenPerceptronIndex]);
            else if(DERIVATIVE == TANH)
                hiddenError[hiddenPerceptronIndex] = error * tanhDerivative(hiddenActivation[hiddenPerceptronIndex]);
        }

        // Update weights for input to hidden layer
        for (int inputPerceptronIndex = 0; inputPerceptronIndex < INPUT_SIZE; inputPerceptronIndex++) {
            for (int hiddenPerceptronIndex = 0; hiddenPerceptronIndex < HIDDEN_SIZE; hiddenPerceptronIndex++) {
                inputToHiddenWeights[inputPerceptronIndex][hiddenPerceptronIndex] += learningRate * hiddenError[hiddenPerceptronIndex] * input[inputPerceptronIndex]; // Ensure input is not normalized again if it was already normalized before calling this method
            }
        }

        // Update biases for hidden and output layers
        for (int hiddenIndex = 0; hiddenIndex < HIDDEN_SIZE; hiddenIndex++) {
            hiddenBias[hiddenIndex] += learningRate * hiddenError[hiddenIndex];
        }

        for (int outputIndex = 0; outputIndex < OUTPUT_SIZE; outputIndex++) {
            outputBias[outputIndex] += learningRate * outputError[outputIndex];
        }

    }
    
    public int predict(double[] input) {
        // This function predicts the label for the input
        // It returns the label with the highest probability

        double[] outputProbabilities = forwardPropagation(input,false);
        int prediction = 0;
        for (int outputIndex = 1; outputIndex < OUTPUT_SIZE; outputIndex++) {
            if (outputProbabilities[outputIndex] > outputProbabilities[prediction]) {
                prediction = outputIndex;//Get the index of the output with the highest probability
            }
        }
        return prediction;//Return the index of the output with the highest probability
    }
    private double sigmoid(double number) {
        return 1 / (1 + Math.exp(-number));
    }

    private double sigmoidDerivative(double number) {
        return sigmoid(number) * (1 - sigmoid(number));// 1/(1+e^-x) * (1-1/(1+e^-x))
    }
    private double relu(double number) {
        return Math.max(0, number);
    }

    private double tanh(double number) {
        return Math.tanh(number);
    }

    private double tanhDerivative(double number) {
        return 1 - Math.pow(tanh(number), POWER_OF_2);
    }
    

    public static void main(String[] args) {
        // This is the main function
        // It runs the MultiLayerPerceptron model on the two datasets and prints the accuracy for each dataset and the average accuracy
        
        System.out.println("Fold 1:");//Print the title for the first fold
        MultiLayerPerceptron model1 = train(PATH_TO_CSV_DATASET1);//Train the model
        System.out.println("Fold 2:");//Print the title for the second fold
        MultiLayerPerceptron model2 = train(PATH_TO_CSV_DATASET2);//Train the model
        System.out.println("First Fold Results:");
        double accuracy1 = test(model1, PATH_TO_CSV_DATASET2);//Test the model on the second dataset
        System.out.println("Second Fold Results:");
        double accuracy2 = test(model2, PATH_TO_CSV_DATASET1);//Test the model on the first dataset
        
        System.out.println("Accuracy for 1st Fold: " + accuracy1+"%");//Print the accuracy for the first dataset
        System.out.println("Accuracy for 2nd Fold: " + accuracy2+"%");//Print the accuracy for the second dataset
        System.out.println("Average Accuracy: " + (accuracy1 + accuracy2) / TWO_FOR_AVG+"%");//Print the average accuracy
    }

    static MultiLayerPerceptron train(String pathToTrainData){
        // This function trains the model using the inputs and targets
        // It returns the trained model

        MultiLayerPerceptron mlp = new MultiLayerPerceptron();//Create a new MultiLayerPerceptron model
        int[][] trainData = readCSV(pathToTrainData);//Read the training data from the csv file

        double[][] inputs = new double[trainData.length][PIXEL_COUNT];//Inputs for the model
        int[] targets = new int[trainData.length];//Targets for the model
        for(int rowIterator = 0; rowIterator < inputs.length; rowIterator++){
            for(int colIterator = 0; colIterator < inputs[0].length; colIterator++){
                inputs[rowIterator][colIterator] = trainData[rowIterator][colIterator]/PIXEL_VALUE;//Normalise the pixel values
                
            }
            targets[rowIterator] = trainData[rowIterator][PIXEL_COUNT];//Get the target label
        }
        mlp.train(inputs, targets, EPOCHS);//Train the model
        return mlp;//Return the trained model
    }

    static double test(MultiLayerPerceptron mlp, String pathToTestData) {
        // This function tests the model on the test data
        // It prints the confusion matrix
        // It returns the accuracy of the model on the test data

        int[][] testData = readCSV(pathToTestData);//Read the test data from the csv file
        double[][] testInputs = new double[testData.length][PIXEL_COUNT];//Inputs for the model
        int[] testTargets = new int[testData.length];//Targets for the model
        for(int rowIterator = 0; rowIterator < testInputs.length; rowIterator++){
            for(int colIterator = 0; colIterator < testInputs[0].length; colIterator++){
                testInputs[rowIterator][colIterator] = testData[rowIterator][colIterator]/PIXEL_VALUE;//Normalise the pixel values
                
            }
            testTargets[rowIterator] = testData[rowIterator][PIXEL_COUNT];//Get the target label
        }
        return getAccuracy(mlp, testInputs, testTargets);//Return the accuracy
    }

    static double getAccuracy(MultiLayerPerceptron mlp, double[][] testInputs, int[] testTargets) {
        // This function calculates the accuracy of the model on the test data
        // It prints the confusion matrix and the misclassified rate for each digit
        // It returns the accuracy of the model on the test data
        
        int correct = 0;//Number of correct predictions
        int[][] confusionMatrix = new int[NUMBER_OF_CLASSES][NUMBER_OF_CLASSES];//Confusion matrix
        for (int targetIterator = 0; targetIterator < testInputs.length; targetIterator++) {
            int prediction = mlp.predict(testInputs[targetIterator]);//Get the prediction
            if (prediction == testTargets[targetIterator]) {//If the prediction is correct
                correct++;//Increment the number of correct predictions
            }
            confusionMatrix[testTargets[targetIterator]][prediction]++;//Increment the confusion matrix
        }
        printConfusionMatrix(confusionMatrix);//Print the confusion matrix
        printMisclassified(confusionMatrix);//Print the misclassified rate for each digit
        return get4Point(correct , testInputs.length);//Return the accuracy
    }

    static void printConfusionMatrix(int[][] confusionMatrix) {
		// This function prints the confusion matrix
        // The confusion matrix is a 2D array where the rows represent the actual labels and the columns represent the predicted labels
        // The value at the intersection of the row and column is the number of images with the actual label and the predicted label

		System.out.println("Confusion Matrix:");
		System.out.print("Label\t");
		for(int predictedLabel = 0; predictedLabel < NUMBER_OF_CLASSES; predictedLabel++) {
			System.out.print(predictedLabel + "\t");//Print the predicted labels at the top
		}
		System.out.println();//Move to the next line

		for(int label = 0; label < NUMBER_OF_CLASSES; label++) {//For each label
			System.out.print(label + "\t");//Print the actual label on the left
			for(int predictedLabel = 0; predictedLabel < NUMBER_OF_CLASSES; predictedLabel++) {//For each predicted label
				System.out.print(confusionMatrix[label][predictedLabel] + "\t");//Print the number of images with the actual label and the predicted label
			}
			System.out.println();//Move to the next line
		}
    }

    static void printMisclassified(int[][] confusionMatrix) {
        // This function prints the misclassified rate for each digit
        // The misclassified rate is the number of misclassified images for each digit as a percentage of the total number of images for that digit
        
        System.out.println("Digit\tMisclassified\tRate");//Print the header for the misclassified rate
        for (int rowIterator = 0; rowIterator < confusionMatrix.length; rowIterator++) {
            int misclassified = 0;//Initialise the number of misclassified images for each digit to 0
            for (int colIterator = 0; colIterator < confusionMatrix[rowIterator].length; colIterator++) {
                if (rowIterator != colIterator) {//if the actual label is not equal to the predicted label
                    misclassified += confusionMatrix[rowIterator][colIterator];//Increment the number of misclassified images for each digit 
                }
            }
            //Print the misclassified rate for each digit
            System.out.println(rowIterator + "\t" + misclassified+ "\t\t" + get4Point(misclassified,(confusionMatrix[rowIterator][rowIterator]+misclassified))+"%");
        }
	}

    static double get4Point(double numerator, double denominator){
        // This function returns the value of a number rounded to 4 decimal places as a percentage
        // The value is calculated as (numerator/denominator)*100 and rounded to 4 decimal places

        return Math.round(numerator*ROUNDING_4_PLACES*CENTURY_FOR_PERCENTAGE/denominator)/ROUNDING_4_PLACES;
    }

    static int[][] readCSV(String filename) {
        // This function reads a csv file and returns a 2D array of integers containing the data from the csv file
        // If there is an error reading the file it prints the stack trace and exits the program

        try{
            File myFile = new File(filename);//creates a new file instance
            int lineCount = countLines(myFile);
            int[][] myArray = new int[lineCount][ROW_LENGTH];//Create an array to store the data from the csv file. Each row has 65 columns(64 pixels + 1 label)
            Scanner readMyFile = new Scanner(myFile);//creates a new scanner instance
            for (int image = 0; image < lineCount; image++) {
                String[] line = readMyFile.nextLine().split(",");
                for (int pixel = 0; pixel < ROW_LENGTH; pixel++) {
                    myArray[image][pixel] = Integer.parseInt(line[pixel]);
                }
            }
            readMyFile.close();
            return myArray;
        } catch (FileNotFoundException invalidPathToFile) {
            System.out.println("File not found at path: "+filename);
            invalidPathToFile.printStackTrace();
            System.exit(0);
        }
        return null;//just to satisfy the compiler
    }

    static int countLines(File myFile) {
        // This function counts the number of lines in a file
        // If there is an error reading the file it prints the stack trace and exits the program

        try{
            Scanner readMyFile = new Scanner(myFile);
            int lineCount = 0;
            while(readMyFile.hasNextLine()) {
                lineCount++;
                readMyFile.nextLine();
            }
            readMyFile.close();
            return lineCount;
        } catch (FileNotFoundException invalidPathToFile) {
            System.out.println("File not found at path: "+myFile.getPath());
            invalidPathToFile.printStackTrace();
            System.exit(0);
        }
        return 0;//just to satisfy the compiler
    }
}