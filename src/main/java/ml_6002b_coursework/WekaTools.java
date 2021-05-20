package ml_6002b_coursework;

import fileIO.OutFile;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.util.Collections;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class WekaTools {
    static Instances data;

    //CONFUSION MATRIX
    public static int[][] confusionMatrix(int[] predicted, int[] actual){
        int[][] confusionMatrix = new int[predicted.length][actual.length];

        for (int i = 0; i < predicted.length; i++) {
            for (int j = 0; j < actual.length; j++) {
                int prediction = predicted[i];
                int actualValue = actual[i];
                if (prediction == actualValue){
                    confusionMatrix[actualValue][prediction]++;
                }

            }
        }
/*
        public int[][] confusionMatrix(int[] predicted, int[] actual){
            int[][] confusionMatrix = new int[2][2];
            for (int i =0; i < predicted.length; i++) {
                confusionMatrix[actual[i]][predicted[i]]++;
            }
            return confusionMatrix;
        }*/


        // Print ConfusionMatrix
        for (int i = 0; i< predicted.length; i++) {
            for (int j =0; j < actual.length; j++) {
                System.out.print(confusionMatrix[i][j] + "\t");
            }
            System.out.println();
        }

        return confusionMatrix;
    }

    //LOAD DATA
    public static Instances loadClassificationData(String fullPath){
        try{
            FileReader reader = new FileReader(fullPath);
            data = new Instances(reader);
        }
        catch(Exception e){
            System.out.println("Exception caught: "+e);
        }
        int numInstances = data.numInstances();
        int numAttributes = data.numAttributes();
        data.setClassIndex(numAttributes-1);

        return data;
    }

    //SPLIT DATA
   public static Instances[] splitData(Instances all, double proportion){
        int numInstances = all.numInstances();
        Instances[] split = new Instances[2];

        split[0]=new Instances(all); // Train
        split[1]=new Instances(all,0); // Test

        int splitIndex = (int)Math.round(proportion * (double)numInstances);

        Collections.shuffle(all, new Random());

        for(int i = 0; i < splitIndex; i++)
        {
            Instance instanceToMove = split[0].remove(0);
            split[1].add(instanceToMove);
        }

        return split;
    }

/*    public static Instances[] splitData(Instances all, double proportion){
        Random r = new Random(System.currentTimeMillis());
        all.randomize(r);
        Instances[] split = new Instances[2];
        split[0] = new Instances(all);
        split[1] = new Instances(all, 0);
        int n = (int) Math.round(proportion * all.numInstances());

        while(split[0].numInstances() != n){
            split[1].add(split[0].get(n));
            split[0].remove(n);
        }
        return split;
    }*/

    //MEASURE ACCURACY
    public static double accuracy(Classifier c, Instances test) throws Exception {
        double count = 0;
        double correct = 0;

        for(Instance instance : test){
            double result = c.classifyInstance(instance);
            double actual = instance.classValue();

            if(result == actual){
                correct++;
            }
            count++;
        }
        //The accuracy of the classifier is then the number correct divided by the number of instances
        double accuracy = ((correct/count)*100);
        return accuracy;
    }

    //CLASS DISTRIBUTION
    public static double[] classDistribution(Instances data){
        double[] classDistribution = new double[data.numClasses()];


        for(int i = 0; i < data.numInstances(); i++)
        {
            classDistribution[(int)data.instance(i).value(data.numAttributes() - 1)]++;
        }

        return classDistribution;
    }

    //GET ALL PREDICTED CLASS VALUES
    public static int[] classifyInstances(Classifier c, Instances test) throws Exception {
        int[] allPredictedClassValues = new int[test.size()];
        for (int i=0; i<test.size(); i++){
            Instance instance = test.instance(i);
            int index = (int) c.classifyInstance(instance); //classifyInstance returns index of predicted
            //TODO: MAYBE add logic to get value from this index

            allPredictedClassValues[i] = index;
        }
        return allPredictedClassValues;
    }

    //GET ALL ACTUAL CLASS VALUES
    public static int[] getClassValues(Instances data){
        int[] allActualClassValues = new int[data.size()];
        for(int i=0; i<data.size(); i++){
            Instance instance = data.instance(i);
            int classValue = (int) instance.classValue();
            allActualClassValues[i] = classValue;
        }
        return allActualClassValues;
    }


    /**
     * Function to generate the test results of a classifier given a train and
     * test dataset.
     *
     * @param classifier to build and test on
     * @param train data
     * @param test data
     * @param outputPath full path to output directory
     * @param outputFile file name of output file (not including extension)
     */
    public static void generateTestResults(Classifier classifier, Instances train, Instances test, String outputPath, String outputFile) throws Exception {
        classifier.buildClassifier(train);

        // setup output file
        OutFile out = new OutFile(outputPath + outputFile + ".csv");
        out.writeLine(train.relationName() + "," + classifier.getClass().getSimpleName());
        out.writeLine("No Parameter Info");
        out.writeLine(String.valueOf(WekaTools.accuracy(classifier, test)));

        // for each instance in test
        for (Instance ins : test) {
            // get predicted class and probabilities of each class
            int prediction = (int) classifier.classifyInstance(ins);
            double[] probabilities = classifier.distributionForInstance(ins);

            // write actual class and predicted class
            out.writeString((int) ins.classValue() + "," + prediction + ",,");

            // write probabilities of each class
            StringBuilder line = new StringBuilder();
            for (double d : probabilities) {
                line.append(d).append(",");
            }
            line.deleteCharAt(line.length() - 1); // remove tailing ','
            out.writeLine(line.toString());
        }
    }

    public static void main(String[] args) throws Exception {
        /*MajorityClassClassifier mc = new MajorityClassClassifier();
        Instances currentData = loadClassificationData("C:\\Users\\" +
                "jpebr\\Desktop\\tsml-master\\tsml\\src\\main\\java\\lab_sheets\\Lab1\\Arsenal_Train.arff");

        //Generate a train/test split
        Instances[] splitArray = splitData(currentData, 0.5);
        Instances train = splitArray[0];
        Instances test = splitArray[1];
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);

        //Build classifier with train data
        mc.buildClassifier(train);
        //Calculate the accuracy
        System.out.println(accuracy(mc, test));
        //Generate the confusion matrix for a single test split (all the data should be in one column).
        int[] predicted = classifyInstances(mc,test);
        int[] actual = getClassValues(train);
        confusionMatrix(predicted, actual);*/

    }

}

