package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

import static ml_6002b_coursework.WekaTools.loadClassificationData;

public class GiniAttributeSplitMeasure implements AttributeSplitMeasure{
    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        double giniIndex = 0, total = 0;

        Instances[] splitData;

        if (att.isNominal()) {
            splitData = splitData(data, att);
        }
        else {
            splitData = splitDataOnNumeric(data, att);
        }

        if (att.isNominal()) {
            for (int i = 0; i < att.numValues(); i++) {
                if (splitData[i].numInstances() > 0) {
                    total += splitData[i].sumOfWeights();
                }
            }
        }
        else {
            for(int i = 0; i < splitData.length; i++){
                if (splitData[i].numInstances() > 0) {
                    total += splitData[i].sumOfWeights();
                }
            }
        }

        for (int i = 0; i < splitData.length; i++) {
            //giniIndex = 0;
            if (splitData[i].numInstances() > 0) {
                giniIndex += (computeGini(splitData[i]) * ((double) splitData[i].numInstances() / total));
            }
        }

        return giniIndex;
    }

    private double computeGini(Instances data) throws Exception {
        double impurity=0;
        double [] classCounts = new double[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }
        for (int j = 0; j < data.numClasses(); j++) {
            if (classCounts[j] > 0) {
                impurity += Math.pow((classCounts[j]/(double)data.numInstances()),2);
            }
        }
        double returnValue = (1 - impurity);
        return returnValue;
    }

    public static void main(String[] args) throws Exception {
/*        Instances currentData = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/Diagnosis_TRAIN.arff");
        Enumeration enumeration = currentData.enumerateAttributes();
        GiniAttributeSplitMeasure giniAttributeSplitMeasure = new GiniAttributeSplitMeasure();
        while(enumeration.hasMoreElements()){
            Attribute att = (Attribute) enumeration.nextElement();
            System.out.println("measure Gini for attribute " + att.name() + " splitting diagnosis = " + giniAttributeSplitMeasure.computeAttributeQuality(currentData, att));
        }*/

        Instances continuousData = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/Chinatown_TRAIN.arff");
        Enumeration continuousEnumeration = continuousData.enumerateAttributes();
        IGAttributeSplitMeasure ig2 = new IGAttributeSplitMeasure();
        while(continuousEnumeration.hasMoreElements()){
            Attribute att = (Attribute) continuousEnumeration.nextElement();
            System.out.println("measure Info Gain for attribute " + att.name() + " splitting diagnosis = " + ig2.computeAttributeQuality(continuousData, att));
        }
    }
}
