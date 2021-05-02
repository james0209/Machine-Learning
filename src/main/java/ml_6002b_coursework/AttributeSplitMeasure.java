package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;

/**
 * Interface for alternative attribute split measures for Part 2.2 of the coursework
 *
 */
public interface AttributeSplitMeasure {

    double computeAttributeQuality(Instances data, Attribute att) throws Exception;

    /**
     * Splits a dataset according to the values of a nominal attribute.
     * Splits Instances by Attribute
     *
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
     default Instances[] splitData(Instances data, Attribute att) {
         if (att.isNominal()) {
             Instances[] splitData = new Instances[att.numValues()];
             for (int j = 0; j < att.numValues(); j++) {
                 splitData[j] = new Instances(data, data.numInstances());
             }
             Enumeration instEnum = data.enumerateInstances();
             while (instEnum.hasMoreElements()) {
                 Instance inst = (Instance) instEnum.nextElement();
                 splitData[(int) inst.value(att)].add(inst);
             }
             for (int i = 0; i < splitData.length; i++) {
                 splitData[i].compactify();
             }

             return splitData;
         }
         else{
             return splitDataOnNumeric(data,att);
         }
    }

    /**
     * Splits a dataset into 2 boolean instances according to the values of a nominal attribute
     *
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
    default Instances[] splitDataOnNumeric(Instances data, Attribute att){
        //System.out.println("Att num values: " + att.numValues());
        Instances[] splitData = new Instances[2];
        splitData[0] = new Instances(data, 0); //Above
        splitData[1] = new Instances(data, 0); //Below

        if(att.isNumeric()){
            AttributeStats as = data.attributeStats(att.index());

            double max = as.numericStats.max;
            double min = as.numericStats.min;
            double random = ((Math.random() * (max - min)) + min);
            System.out.println("Max: " + max + " Min: " + min + " Random: " + random);

            for(int i = 0; i < data.numInstances(); i++)
            {
                Instance instanceToCheck = data.instance(i);
                if (instanceToCheck.value(att) > random) {
                    splitData[0].add(instanceToCheck);
                }
                else {
                    splitData[1].add(instanceToCheck);
                }
            }
            for (Instances splitDatum : splitData) {
                splitDatum.compactify();
            }

/*            for (Instance x : splitData[0]){
                System.out.println("Above: " + x.value(att));
            }
            for (Instance x : splitData[1]){
                System.out.println("Below: " + x.value(att));
            }*/
            return splitData;

        }

        else {
            System.out.println("Attribute is Nominal");
            return splitData(data,att);
        }


    }

}
