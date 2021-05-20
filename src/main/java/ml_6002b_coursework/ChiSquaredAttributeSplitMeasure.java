package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

import static ml_6002b_coursework.AttributeMeasures.measureChiSquared;
import static ml_6002b_coursework.AttributeMeasures.measureChiSquaredYates;
import static ml_6002b_coursework.WekaTools.loadClassificationData;

public class ChiSquaredAttributeSplitMeasure implements AttributeSplitMeasure {
    /**
     * Implement and test a class ChiSquaredAttributeSplitMeasure that implements
     * AttributeSplitMeasure and measures the quality using the chi-squared statistic. This
     * class should be configurable to use the Yates correction.
     **/

    boolean yates = false;

    public boolean isYates() {
        return yates;
    }

    public ChiSquaredAttributeSplitMeasure() {
    }

    public ChiSquaredAttributeSplitMeasure(boolean yates) {
        this.yates = yates;
    }


    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
/*        if (data.numInstances() == 0)
            return 0;*/

        Instances[] splitData;
        int numValues;

        if(att.isNominal()){
            splitData = splitData(data, att);
            numValues = att.numValues();
        }
        else{
            splitData = splitDataOnNumeric(data, att).getKey();
            numValues = splitData.length;
        }

        //System.out.println("num values" + numValues);
        //System.out.println("num classes" + data.numClasses());

        int[][] table = new int[numValues][data.numClasses()];
        for (int i = 0; i < numValues; i++) {
            for (Instance instance : splitData[i]) {
                int value = (int) instance.classValue();
                table[i][value]++;
            }
        }

        if(yates){
            return measureChiSquaredYates(table);
        }
        else{
            return measureChiSquared(table);
        }
    }

    public static void main(String[] args) throws Exception {
        Instances currentData = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/Diagnosis_TRAIN.arff");
        Instances chinatownTrain = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/Chinatown_TRAIN.arff");
        Instances chinatownTest = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/Chinatown_TEST.arff");

        Enumeration enumeration = currentData.enumerateAttributes();
        ChiSquaredAttributeSplitMeasure chiSquaredAttributeSplitMeasure = new ChiSquaredAttributeSplitMeasure(false);
        ChiSquaredAttributeSplitMeasure yates = new ChiSquaredAttributeSplitMeasure(true);
        while(enumeration.hasMoreElements()){
            Attribute att = (Attribute) enumeration.nextElement();
            System.out.println("measure Chi for attribute " + att.name() + " splitting diagnosis = " +
                    chiSquaredAttributeSplitMeasure.computeAttributeQuality(currentData, att));
            System.out.println("measure Chi Yates for attribute " + att.name() + " splitting diagnosis = " +
                    yates.computeAttributeQuality(currentData, att));

        }

/*        Enumeration enumeration2 = chinatownTrain.enumerateAttributes();
        ChiSquaredAttributeSplitMeasure chiSquaredAttributeSplitMeasure = new ChiSquaredAttributeSplitMeasure(false);
        ChiSquaredAttributeSplitMeasure yates = new ChiSquaredAttributeSplitMeasure(true);
        while(enumeration2.hasMoreElements()){
            Attribute att = (Attribute) enumeration2.nextElement();
            System.out.println("measure Chi for attribute " + att.name() + " splitting diagnosis = " +
                    chiSquaredAttributeSplitMeasure.computeAttributeQuality(chinatownTrain, att));
            System.out.println("measure Chi Yates for attribute " + att.name() + " splitting diagnosis = " +
                    yates.computeAttributeQuality(chinatownTrain, att));

        }*/
    }
}
