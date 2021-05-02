package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

import static ml_6002b_coursework.WekaTools.loadClassificationData;

public class IGAttributeSplitMeasure implements AttributeSplitMeasure{

    /**
     * Computes information gain for an attribute.
     *
     * @param data the data for which info gain is to be computed
     * @param att the attribute
     * @return the information gain for the given attribute and data
     * @throws Exception if computation fails
     */
    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        double infoGain = computeEntropy(data);
        //double infoGain = 0;
        Instances[] splitData;

        if (att.isNominal()) {
            splitData = splitData(data, att);
        }
        else {
            splitData = splitDataOnNumeric(data, att);
        }



        if (att.isNominal()){
            for (int j = 0; j < att.numValues(); j++) {
                if (splitData[j].numInstances() > 0) {
                    infoGain -= ((double) splitData[j].numInstances() /
                            (double) data.numInstances()) *
                            computeEntropy(splitData[j]);
                }
            }
            return infoGain;
        }
        else{
            for(int i = 0; i < splitData.length; i++){
                if (splitData[i].numInstances() > 0) {
                    infoGain -= ((double) splitData[i].numInstances() /
                            (double) data.numInstances()) *
                            computeEntropy(splitData[i]);
                }
            }
            return infoGain;
        }


    }

    /**
     * Computes the entropy of a dataset.
     *
     * @param data the data for which entropy is to be computed
     * @return the entropy of the data's class distribution
     * @throws Exception if computation fails
     */
    private double computeEntropy(Instances data) throws Exception {

        double [] classCounts = new double[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }
        double entropy = 0;
        for (int j = 0; j < data.numClasses(); j++) {
            if (classCounts[j] > 0) {
                entropy -= classCounts[j] * Utils.log2(classCounts[j]);
            }
        }
        entropy /= (double) data.numInstances();
        return entropy + Utils.log2(data.numInstances());
    }

    public static void main(String[] args) throws Exception {
        Instances currentData = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/Diagnosis_TRAIN.arff");
        Enumeration enumeration = currentData.enumerateAttributes();
        IGAttributeSplitMeasure ig = new IGAttributeSplitMeasure();
        while(enumeration.hasMoreElements()){
            Attribute att = (Attribute) enumeration.nextElement();
            System.out.println("measure Info Gain for attribute " + att.name() + " splitting diagnosis = " + ig.computeAttributeQuality(currentData, att));
        }

        Instances continuousData = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/optdigits.arff");
        Enumeration continuousEnumeration = continuousData.enumerateAttributes();
        IGAttributeSplitMeasure ig2 = new IGAttributeSplitMeasure();
        while(continuousEnumeration.hasMoreElements()){
            Attribute att = (Attribute) continuousEnumeration.nextElement();
            System.out.println("measure Info Gain for attribute " + att.name() + " splitting diagnosis = " + ig2.computeAttributeQuality(continuousData, att));
        }
    }


}
