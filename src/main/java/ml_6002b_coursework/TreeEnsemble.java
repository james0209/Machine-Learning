package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class TreeEnsemble extends AbstractClassifier{
    @Override
    public void buildClassifier(Instances data) throws Exception {

    }


    @Override
    public double classifyInstance(Instance instance)
    {
        return 0.0;
    }


    @Override
    public double[] distributionForInstance(Instance ins){

    }

    public static void main(String[] args) throws Exception {

    }
}
