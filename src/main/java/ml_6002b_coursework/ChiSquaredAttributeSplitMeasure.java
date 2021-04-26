package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

import static ml_6002b_coursework.WekaTools.loadClassificationData;

//TODO: Measure the quality using the chi-squared statistic
//TODO: Make this configurable to use Yates correction

public class ChiSquaredAttributeSplitMeasure implements AttributeSplitMeasure{
    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        /**
         * For each observed number in the table subtract the corresponding expected number
         * Square the difference [ (O —E)2 ].
         * Divide the squares obtained for each cell in the table by the expected number for that cell [ (O - E)2 / E ].
         * Sum all the values for (O - E)2 / E. This is the chi square statistic.
         */

        boolean yates = true;

        double chiValue = 0;
        double total = 0;

        Instances[] splitData;

        if (att.isNominal()) {
            splitData = splitData(data, att);
        }
        else {
            splitData = splitDataOnNumeric(data, att);
        }


        for (int i = 0; i < att.numValues(); i++) {
            if (splitData[i].numInstances() > 0) {
                total += splitData[i].sumOfWeights();
            }
        }

        for (int i = 0; i < att.numValues(); i++) {
            //giniIndex = 0;
            if (splitData[i].numInstances() > 0) {
                chiValue += (computeChiSquared(splitData[i], att) * ((double) splitData[i].numInstances() / total));
            }
        }

        int numInstances = data.numInstances();
        int numClasses = att.numValues();

        return chiValue;
    }

    /**
     * Computes the chi squared value of a dataset.
     *
     * @param data the data for which chi squared is to be computed
     * @return the chi square of the data's class distribution
     * @throws Exception if computation fails
     */
    private double computeChiSquared(Instances data, Attribute att) throws Exception {
        double [] classCounts = new double[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }

        double chiSquared = 0;
        double[] expected = new double[data.numInstances()];

        for (int i = 0; i < data.numClasses(); i++) {
            if (classCounts[i] > 0) {
                //TODO: FIX EXPECTED CALCULATION
            }
        }

        return chiSquared;
    }

    public static void main(String[] args) throws Exception {
        Instances currentData = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/Diagnosis_TRAIN.arff");
        Enumeration enumeration = currentData.enumerateAttributes();
        ChiSquaredAttributeSplitMeasure chiSquaredAttributeSplitMeasure = new ChiSquaredAttributeSplitMeasure();
        while(enumeration.hasMoreElements()){
            Attribute att = (Attribute) enumeration.nextElement();
            System.out.println("measure chi square for attribute " + att.name() + " splitting diagnosis = " + chiSquaredAttributeSplitMeasure.computeAttributeQuality(currentData, att));
        }
    }
}


