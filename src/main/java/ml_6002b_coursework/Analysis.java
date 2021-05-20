package ml_6002b_coursework;

import evaluation.storage.ClassifierResults;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import static ml_6002b_coursework.WekaTools.loadClassificationData;
import static utilities.InstanceTools.resampleInstances;

public class Analysis {

    private static void code() throws Exception {

        String path = "C:\\Users\\jpebr\\Desktop\\tsml-master\\tsml\\src\\main\\java\\ml_6002b_coursework\\test_data\\";
        String fileName = "optdigits";
        //String fileName = "MixedShapesSmallTrain_TRAIN";

        Instances data = WekaTools.loadClassificationData(path + fileName + ".arff");
        //Instances test = WekaTools.loadClassificationData(path + "MixedShapesSmallTrain_TEST" + ".arff");

        Instances[] trainTest = resampleInstances(data, 0, 0.6);

        Instances splitTrain = trainTest[0];
        Instances splitTest = trainTest[1];

        String[] options = new String[1];
        options[0] = "-I";

        /*
         * Classifier
         */
        ID3Coursework classifier = new ID3Coursework();
        classifier.setOptions(options);
        //J48 ensemble = new J48();
        //Id3 ensemble = new Id3();

        //TreeEnsemble ensemble = new TreeEnsemble();
        //RotationForest ensemble = new RotationForest();

        classifier.buildClassifier(splitTrain);
        Evaluation eval = new Evaluation(splitTrain);
        eval.evaluateModel(classifier, splitTest);
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());

        WekaTools.generateTestResults(classifier, splitTrain, splitTest, "C:\\Users\\jpebr\\Desktop\\tsml-master\\tsml\\src\\main\\java\\ml_6002b_coursework\\test_results\\", "classifierTestResults");
        String classifiertestOutput = "classifierTestResults.csv";

        ClassifierResults ensembleres = new ClassifierResults();
        ensembleres.loadResultsFromFile("C:\\Users\\jpebr\\Desktop\\tsml-master\\tsml\\src\\main\\java\\ml_6002b_coursework\\test_results\\" + classifiertestOutput);
        ensembleres.findAllStats();

        System.out.println("Accuracy: " + ensembleres.getAcc());
        System.out.println("Balanced Accuracy: " + ensembleres.balancedAcc);
        System.out.println("Negative Log Likelihood: " + ensembleres.nll);
        System.out.println("Area Under ROC: " + ensembleres.meanAUROC);
    }

    public static void main(String[] args) throws Exception {
        code();

        /*Instances trainingDataSet = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/MixedShapesSmallTrain_TRAIN.arff");
        Instances testingDataSet = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/MixedShapesSmallTrain_TEST.arff");

        for (String s : DatasetLists.nominalAttributeProblems){
            Instances data = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/" + s);
        }

        System.out.println("************************** J48 *************************");
        Classifier classifier = new J48();
        classifier.buildClassifier(trainingDataSet);
        Evaluation eval = new Evaluation(trainingDataSet);
        eval.evaluateModel(classifier, testingDataSet);
        // Print the algorithm summary
        System.out.println("** Decision Tress Evaluation with Datasets **");
        System.out.println(eval.toSummaryString());
        System.out.print(" the expression for the input data as per algorithm is ");
        System.out.println(classifier);
        System.out.println(eval.toMatrixString());
        System.out.println(eval.toClassDetailsString());


        System.out.println("************************** ID3Coursework *************************");
        ID3Coursework id3Classifier = new ID3Coursework();
        String[] options = new String[1];
        options[0] = "-I";
        id3Classifier.setOptions(options);

        id3Classifier.buildClassifier(trainingDataSet);
        Evaluation evalId3 = new Evaluation(trainingDataSet);
        evalId3.evaluateModel(id3Classifier, testingDataSet);
        // Print the algorithm summary
        System.out.println("** Decision Tress Evaluation with Datasets **");
        System.out.println(evalId3.toSummaryString());
        System.out.print(" the expression for the input data as per algorithm is ");
        System.out.println(id3Classifier);
        System.out.println(evalId3.toMatrixString());
        System.out.println(evalId3.toClassDetailsString());
        */
    }
}
