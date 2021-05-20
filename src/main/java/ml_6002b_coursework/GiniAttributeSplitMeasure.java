package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

import static ml_6002b_coursework.WekaTools.loadClassificationData;

public class GiniAttributeSplitMeasure implements AttributeSplitMeasure {

    /**
     * Computes gini index for an attribute.
     *
     * @param data the data for which info gain is to be computed
     * @param att the attribute
     * @return the gini index for the given attribute and data
     */
    @Override
    public double computeAttributeQuality(Instances data, Attribute att) {

        // Gini at root node
        double gini = computeImpurity(data);

        Instances[] splitData;
        int numValues;

        if (att.isNominal()) {
            splitData = splitData(data, att);
            numValues = att.numValues();
        }
        else {
            splitData = splitDataOnNumeric(data, att).getKey();
            numValues = splitData.length;
        }

        for (int j = 0; j < splitData.length; j++) {
            if (splitData[j].numInstances() > 0) {
                gini -= ((double)splitData[j].numInstances() /
                        (double)data.numInstances()) *
                        computeImpurity(splitData[j]);
            }
        }
        return gini;
    }

    /**
     * Computes the impurity of a dataset.
     *
     * @param data the data for which impurity is to be computed
     * @return the impurity of the data's class distribution
     */
    private double computeImpurity(Instances data) {

        double [] counts = new double[data.numClasses()];
        double impurity = 1.0;
        double numInstances = data.numInstances();

        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            counts[(int)inst.classValue()]++;
        }

        for (double classCount : counts) {
            if (classCount > 0) {
                double p = classCount / numInstances;
                impurity -= p * p;
            }
        }

        return impurity;
    }

    public static void main (String[] args) throws Exception {
        Instances currentData = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/Diagnosis_TRAIN.arff");
        Enumeration enumeration = currentData.enumerateAttributes();
        GiniAttributeSplitMeasure giniAttributeSplitMeasure = new GiniAttributeSplitMeasure();
        while(enumeration.hasMoreElements()){
            Attribute att = (Attribute) enumeration.nextElement();
            System.out.println("measure Gini for attribute " + att.name() + " splitting diagnosis = " + giniAttributeSplitMeasure.computeAttributeQuality(currentData, att));
        }
    }
}
