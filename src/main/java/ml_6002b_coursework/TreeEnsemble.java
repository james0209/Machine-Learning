package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class TreeEnsemble extends AbstractClassifier{

    private int ensembleSize = 50;
    double attributeProportion = 0.5;
    ID3Coursework[] ensembleContainer = new ID3Coursework[ensembleSize];

    /**
     * Builds a TreeEnsemble
     * @param data
     * @throws Exception e
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {

        // construct a new set of instances for each element of the ensemble by selecting a random subset
        // (without replacement) of the attributes.
        for (int i = 0; i < this.ensembleSize; i++) {
            // reset copy of instances
            Instances instancesCopy = new Instances(data);
            Collections.shuffle(instancesCopy);

            //Create subset of the instance

        }


        // build a separate classifier on each Instances object

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
