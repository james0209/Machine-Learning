package ml_6002b_coursework;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import static ml_6002b_coursework.WekaTools.loadClassificationData;

public class Analysis {

    public static void main(String[] args) throws Exception {
        Instances trainingDataSet = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/Chinatown_TRAIN.arff");
        Instances testingDataSet = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/Chinatown_TEST.arff");

        for (String s : DatasetLists.nominalAttributeProblems){
            Instances data = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/" + s);
        }


        /**
         * train the alogorithm with the training data and evaluate the
         * algorithm with testing data
         */
        System.out.println("************************** J48 *************************");
        /** Classifier here is Linear Regression */
        Classifier classifier = new J48();
        classifier.buildClassifier(trainingDataSet);
        Evaluation eval = new Evaluation(trainingDataSet);
        eval.evaluateModel(classifier, testingDataSet);
        /** Print the algorithm summary */
        System.out.println("** Decision Tress Evaluation with Datasets **");
        System.out.println(eval.toSummaryString());
        System.out.print(" the expression for the input data as per algorithm is ");
        System.out.println(classifier);
        System.out.println(eval.toMatrixString());
        System.out.println(eval.toClassDetailsString());


        System.out.println("************************** ID3 *************************");
        /** Classifier here is Linear Regression */
        Classifier id3Classifier = new Id3();

        //J48,Id3
        /** */
        id3Classifier.buildClassifier(trainingDataSet);
        /**
         * train the alogorithm with the training data and evaluate the
         * algorithm with testing data
         */
        Evaluation evalId3 = new Evaluation(trainingDataSet);
        evalId3.evaluateModel(id3Classifier, testingDataSet);
        /** Print the algorithm summary */
        System.out.println("** Decision Tress Evaluation with Datasets **");
        System.out.println(evalId3.toSummaryString());
        System.out.print(" the expression for the input data as per algorithm is ");
        System.out.println(id3Classifier);
        System.out.println(evalId3.toMatrixString());
        System.out.println(evalId3.toClassDetailsString());
    }
}
