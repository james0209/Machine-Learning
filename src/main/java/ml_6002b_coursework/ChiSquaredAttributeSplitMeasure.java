package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instances;

//TODO: Measure the quality using the chi-squared statistic
//TODO: Make this configurable to use Yates correction

public class ChiSquaredAttributeSplitMeasure implements AttributeSplitMeasure{
    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        return 0;
    }
}
