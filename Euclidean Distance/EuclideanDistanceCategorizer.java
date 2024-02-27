import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class EuclideanDistanceCategorizer {
    // This class implements a Euclidean Distance Categorizer to classify images of handwritten digits
    // There are 2 datasets of handwritten digits and their labels
    // We do a 2 fold test for the algorithm using the 2 datasets
    // This program prints the confusion matrix and the misclassified rate for each digit for each fold
    // It prints the average accuracy for the algorithm on the two folds.

    private static final String PATH_TO_CSV_DATASET1 = "cw2DataSet1.csv";//Path to the first csv file
    private static final String PATH_TO_CSV_DATASET2 = "cw2DataSet2.csv";//Path to the second csv file
    private static final int PIXELS_IN_ROW = 8;//8 pixels in a row
    private static final int PIXEL_COUNT = PIXELS_IN_ROW*PIXELS_IN_ROW;//8x8 pixels in the image = 64 pixels
    private static final int ROW_LENGTH = PIXEL_COUNT+1;//64 pixels + 1 label in the csv file for each row = 65 columns
    private static final int POWER_OF_TWO = 2;//Used to calculate the euclidean distance
    private static final int NUMBER_OF_CATEGORIES = 10;//Number of categories in the data set (0-9)
    private static final int TWO_FOR_AVG = 2;//Used to calculate the average accuracy
    private static final double CENTURY_FOR_PERCENTAGE = 100.0;//This is used to calculate the percentage
    private static final double ROUNDING_4_PLACES = 10000.0;//This is used to round the number to 4 decimal places
    
    public static void main(String[] args) {
        //This is the main function
        //It reads the two csv files and runs the Euclidean distance algorithm on them
        //It prints the accuracy of the algorithm

        //Read the csv files
        int[][] dataSet1 = readCSV(PATH_TO_CSV_DATASET1);
        int[][] dataSet2 = readCSV(PATH_TO_CSV_DATASET2);
        
        //Run the Euclidean distance algorithm on the two datasets
        System.out.println("Categorizing by Euclidean distance");
        System.out.println("1st Fold");
        double accuracy1 = runEuclideanDistance(dataSet1, dataSet2);
        System.out.println("2nd Fold");
        double accuracy2 = runEuclideanDistance(dataSet2, dataSet1);

        //Print the accuracies of the algorithm
        System.out.println("Accuracy for 1st Fold: " + accuracy1+"%");
        System.out.println("Accuracy for 2nd Fold: " + accuracy2+"%");
        System.out.println("Average Accuracy: " + (accuracy1 + accuracy2) / TWO_FOR_AVG+"%");
    }

    static double runEuclideanDistance(int[][] dataSet1, int[][] dataSet2){
        // This function runs the Euclidean distance algorithm
        // It categorizes the first data set based on the second data set
        // It prints the confusion matrix and the misclassified rate for each digit
        // It returns the accuracy of the algorithm

        int[] predictedLabels = new int[dataSet1.length];
        for (int outerLoopIterator = 0; outerLoopIterator < dataSet1.length; outerLoopIterator++) {
            int minIndex = 0;//Initialise the Index of the image with the minimum distance to the first image in the second data set
            double minDistance = euclideanDistance(dataSet1[outerLoopIterator],dataSet2[minIndex]);//Initialise the minimum distance
            for (int innerLoopIterator = 1; innerLoopIterator < dataSet2.length; innerLoopIterator++) {//For each image in the second data set
                double distance = euclideanDistance(dataSet1[outerLoopIterator], dataSet2[innerLoopIterator]);//Calculate the euclidean distance between the two images
                if (distance < minDistance) {
                    minDistance = distance;//Store the minimum distance
                    minIndex = innerLoopIterator;//Store the index of the image with the minimum distance
                }
            }
            predictedLabels[outerLoopIterator] = dataSet2[minIndex][PIXEL_COUNT];//Predict the label of the image
        }

        //Test vs Actual
        int correct = 0;//Count of correct predictions
        int[][] confusionMatrix = new int[NUMBER_OF_CATEGORIES][NUMBER_OF_CATEGORIES];//Create a 2D array to store the confusion matrix
        for (int image = 0; image < dataSet1.length; image++) {//For each image in the data set
            if (predictedLabels[image] == dataSet1[image][PIXEL_COUNT]) {//If the predicted label is the same as the actual label
                correct++;//Increment the correct count
            }
            confusionMatrix[dataSet1[image][PIXEL_COUNT]][predictedLabels[image]]++;//Increment the confusion matrix
        }
        printConfusionMatrix(confusionMatrix);//Print the confusion matrix
        printMisclassified(confusionMatrix);//Print the misclassified rate for each digit
        double accuracy = get4Point(correct, dataSet1.length);//Calculate the accuracy
        System.out.println("Accuracy: "+accuracy+"%");//Print the accuracy
        return accuracy;//Return the accuracy
    }

    static double euclideanDistance(int[] vector1, int[] vector2) {
        // This function calculates the euclidean distance between two 64-dimensional vectors
        // It returns the euclidean distance between the two vectors
        // The euclidean distance is calculated as the square root of the sum of the squares of the differences between the corresponding elements of the two vectors
        // The formula is: sqrt((x1-y1)^2 + (x2-y2)^2 + ... + (x64-y64)^2) for two 64-dimensional vectors (x1,x2,...,x64) and (y1,y2,...,y64)

        double sum = 0;
        for (int index = 0; index < PIXEL_COUNT; index++) {
            sum += Math.pow(vector1[index] - vector2[index], POWER_OF_TWO);
        }
        return Math.sqrt(sum);
    }

    static void printConfusionMatrix(int[][] confusionMatrix) {
		// This function prints the confusion matrix
        // The confusion matrix is a 2D array where the rows represent the actual labels and the columns represent the predicted labels
        // The value at the intersection of the row and column is the number of images with the actual label and the predicted label

		System.out.println("Confusion Matrix:");
		System.out.print("Label\t");
		for(int predictedLabel = 0; predictedLabel < NUMBER_OF_CATEGORIES; predictedLabel++) {
			System.out.print(predictedLabel + "\t");//Print the predicted labels at the top
		}
		System.out.println();//Move to the next line

		for(int label = 0; label < NUMBER_OF_CATEGORIES; label++) {//For each label
			System.out.print(label + "\t");//Print the actual label on the left
			for(int predictedLabel = 0; predictedLabel < NUMBER_OF_CATEGORIES; predictedLabel++) {//For each predicted label
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