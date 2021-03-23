package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

public class TreeEnsemble extends AbstractClassifier{
    private int ensembleSize = 50;
    private ArrayList<C45Coursework> ensembleContainer = new ArrayList<C45Coursework>(ensembleSize);

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
        return null;
    }

    public static void main(String[] args) throws Exception {

    }
}
