package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

import static ml_6002b_coursework.AttributeMeasures.measureChiSquared;
import static ml_6002b_coursework.AttributeMeasures.measureChiSquaredYates;
import static ml_6002b_coursework.WekaTools.loadClassificationData;

public class ChiSquaredAttributeSplitMeasure implements AttributeSplitMeasure {

    boolean yates = false;

    public ChiSquaredAttributeSplitMeasure() {
    }

    public ChiSquaredAttributeSplitMeasure(boolean yates) {
        this.yates = yates;
    }

    /**
     * Computes Chi Squared statistic for an attribute.
     *
     * @param data the data for which chi is to be computed
     * @param att  the attribute
     * @return the chi for the given attribute and data
     */
    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        if (data.numInstances() == 0)
            return 0;

        Instances[] splitData;
        int numValues;

        if (att.isNominal()) {
            splitData = splitData(data, att);
            numValues = att.numValues();
            System.out.println(numValues);
        }
        else {
            splitData = splitDataOnNumeric(data, att).getKey();
            numValues = splitData.length;
        }

        double[][] matrix = new double[numValues][data.numClasses()];
        for (int i = 0; i < numValues; i++) {
            for (Instance instance : splitData[i]) {
                int value = (int) instance.classValue();
                matrix[i][value]++;
            }
        }

        if(yates){
            return measureChiSquaredYates(matrix);
        }
        else{
            return measureChiSquared(matrix);
        }
    }

    @Override
    public String toString() {
        if (yates) {
            return "-Y: Split is Chi Squared statistic with Yates correction.";
        } else {
            return "-C: Split is Chi Squared statistic.";
        }

    }

    public static void main (String[] args) throws Exception {
        Instances currentData = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/optdigits.arff");
        Enumeration enumeration = currentData.enumerateAttributes();
        ChiSquaredAttributeSplitMeasure chiSquaredAttributeSplitMeasure = new ChiSquaredAttributeSplitMeasure();
        ChiSquaredAttributeSplitMeasure yates = new ChiSquaredAttributeSplitMeasure(true);
        while(enumeration.hasMoreElements()){
            Attribute att = (Attribute) enumeration.nextElement();
            System.out.println("measure Chi for attribute " + att.name() + " splitting diagnosis = " +
                    chiSquaredAttributeSplitMeasure.computeAttributeQuality(currentData, att));
            System.out.println("measure Chi Yates for attribute " + att.name() + " splitting diagnosis = " +
                    yates.computeAttributeQuality(currentData, att));

        }
    }
}