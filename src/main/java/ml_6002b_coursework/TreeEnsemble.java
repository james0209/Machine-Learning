package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.*;

import java.util.*;

import static ml_6002b_coursework.WekaTools.loadClassificationData;

public class TreeEnsemble extends AbstractClassifier{

    private int ensembleSize;
    private int numAttributes;
    private double attributeProportion;
    private ID3Coursework[] ensembleContainer;
    private Attribute[][] attributesForEachClassifier;

    /**
     * Default constructor for a TreeEnsemble
     */
    public TreeEnsemble(){
        ensembleSize = 50;
        ensembleContainer = new ID3Coursework[ensembleSize];
        attributeProportion = 0.5;
    }

    /**
     * Builds a TreeEnsemble
     * @param data
     * @throws Exception e
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {

        this.attributesForEachClassifier = new Attribute[ensembleSize][data.numAttributes()];

        for (int i = 0; i < ensembleSize; i++) {
            ID3Coursework classifier = new ID3Coursework();

            ////// SELECT ATTRIBUTES
            int numAttribs = data.numAttributes() - 1;
            int numToSelect = (int)Math.round(numAttributes * attributeProportion);

            // Array of random index's for attributes to remove
            int[] randomIndices = new int[numToSelect];

            // Generate a random array of indices as index's for the attributes
            for (int j = 0; j < numToSelect; j++){
                Random rnd = new Random();
                int randomIndex = rnd.nextInt(numAttribs);
                //TODO: CHECK IF NUMBER IS ALREADY IN ARRAY
                randomIndices[j] = randomIndex;
            }

            // Array of selected attributes
            Attribute[] selected = new Attribute[numToSelect];

            for (int k = 0; k < randomIndices.length; k++){
                int ind = randomIndices[k];
                Attribute select = data.attribute(ind);
                if (select.index() < data.numAttributes()-1) {
                    selected[k] = select;
                }
            }

            attributesForEachClassifier[i] = selected;
            Instances modifiedData = data;

            for (Attribute att : attributesForEachClassifier[i]){
                if (att.index() < data.numAttributes() - 1) {
                    modifiedData.deleteAttributeAt(att.index());
                }
            }

            // Build a separate classifier on each Instances object
            classifier.buildClassifier(modifiedData);
            ensembleContainer[i] = classifier;

        }

    }


    @Override
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
        double predictedClass;
        int[] countVotes = new int[2];

        for (int i = 0; i < ensembleSize; i++) {
            ID3Coursework classifier = ensembleContainer[i];

            Instance modified = instance;
            for (Attribute a : attributesForEachClassifier[i]){
                if (a.index() < instance.numAttributes()-1) {
                    modified.deleteAttributeAt(a.index());
                }
            }

            double classPredicted = classifier.classifyInstance(modified);
            if (classPredicted == 0) {
                countVotes[0]++;
            }
            else {
                countVotes[1]++;
            }
        }

        if (countVotes[0] > countVotes[1]){
            predictedClass = 0;
        }
        else {
            predictedClass = 1;
        }

        return predictedClass;
    }


    @Override
    public double[] distributionForInstance(Instance ins) throws NoSupportForMissingValuesException {
        double[] distribution = new double[2];
        int[] countVotes = new int[2];

        for (int i = 0; i < this.ensembleSize; i++) {
            ID3Coursework id3 = this.ensembleContainer[i];

            Instance modified = ins;
            for (Attribute a : attributesForEachClassifier[i]){
                if (a.index() < ins.numAttributes()-1) {
                    modified.deleteAttributeAt(a.index());
                }
            }

            double classPredicted = id3.classifyInstance(modified);
            if (classPredicted == -1) {
                countVotes[0]++;
            }
            else {
                countVotes[1]++;
            }
        }

        distribution[0] = (double) countVotes[0]/ensembleSize;
        distribution[1] = (double) countVotes[1]/ensembleSize;

        return distribution;
    }

    public static void main(String[] args) throws Exception {
        Instances trainingData = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/optdigits.arff");
        TreeEnsemble treeEnsemble = new TreeEnsemble();

        try
        {
            treeEnsemble.buildClassifier(trainingData);
            trainingData.setClassIndex(trainingData.numAttributes()-1);
        } catch (Exception e)
        {
            e.printStackTrace();
        }

        Instances testData = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/optdigits.arff");
        for (Instance data : testData){
            System.out.println("classify instance = " + treeEnsemble.classifyInstance(data));
        }
/*        System.out.println();
        for (Instance data : testData){
            double[] distribution = treeEnsemble.distributionForInstance(data);
            System.out.print("distribution for instance = ");
            for (double d : distribution){
                System.out.print(d + ", ");
            }
            System.out.println();
        }
        System.out.println("Test Accuracy: ");
        System.out.println(WekaTools.accuracy(treeEnsemble, testData));*/

    }


}
