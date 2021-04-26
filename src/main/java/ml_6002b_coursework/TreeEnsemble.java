package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;

import static ml_6002b_coursework.WekaTools.loadClassificationData;

public class TreeEnsemble extends AbstractClassifier{

    private int ensembleSize;
    private int numAttributes;
    private double attributeProportion;
    private ID3Coursework[] ensembleContainer;
    private boolean[][] attributesForEachClassifier;

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

        numAttributes = data.numAttributes();
        attributesForEachClassifier = new boolean[ensembleSize][numAttributes];

        // copy of the Instances
        Instances instancesCopy = new Instances(data);

        // construct a new set of instances for each element of the ensemble by selecting a random subset
        // (without replacement) of the attributes.
        for (int i = 0; i < ensembleSize; i++) {

            // shuffle instances on every run, but don't reset as there should be no replacement
            Collections.shuffle(instancesCopy);

            int numCopyInstances = instancesCopy.numInstances();
            int numCopyAttributes = instancesCopy.numAttributes();

            //int subsetInstances = (int) Math.round((instancesPercent/100) * numInstances);
            int subsetAttributes = (int)Math.round(numAttributes * attributeProportion);

            Instances subset = new Instances(instancesCopy, 0, subsetAttributes);

            int numAttributesToRemove = (int)Math.ceil(instancesCopy.numAttributes() * (1.0-attributeProportion));

            // Build a separate classifier on each Instances object
            ID3Coursework classifier = new ID3Coursework();
            classifier.buildClassifier(subset);
            ensembleContainer[i] = classifier;

        }

    }


    @Override
    public double classifyInstance(Instance instance)
    {
        // Return the majority vote of the ensemble
        //TODO: Look at whether this actually uses the Votes, or just bases it off of probabilities
        double[] probabilities = distributionForInstance(instance);
        double[] votes = new double[probabilities.length];

        int index = 0;
        double highestProb = 0.0;

        for(int i = 0; i < instance.numClasses(); i++)
        {
            if(probabilities[i] > highestProb)
            {
                highestProb = probabilities[i];
                index = i;
            }
        }

        int c=(int)instance.classValue();
        votes[c]++;
        for (double d : votes){

        }

        return index;
    }


    @Override
    public double[] distributionForInstance(Instance ins){
        double[] probabilities = new double[ins.numClasses()];

        return null;
    }

    public static void main(String[] args) throws Exception {
        Instances currentData = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/optdigits.arff");
        TreeEnsemble treeEnsemble = new TreeEnsemble();

        try
        {
            treeEnsemble.buildClassifier(currentData);
        } catch (Exception e)
        {
            e.printStackTrace();
        }
        System.out.println("Test Accuracy: ");
        System.out.println(WekaTools.accuracy(treeEnsemble, currentData));


    }
}
