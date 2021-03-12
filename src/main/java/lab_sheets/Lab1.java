package lab_sheets;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;

public class Lab1 extends AbstractClassifier {
    int attribute=0;
    double mean = 0;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        double mean = 0;
        for (Instance ins : data){
            mean+=ins.value(0);
        }
        mean/=data.numInstances();
    }

    @Override
    //Classifies the given test instance.
    public double classifyInstance(Instance instance) throws Exception {
        //double[] values = distributionForInstance(instance);
        //Arrays.sort(values);

        double value = instance.value(attribute);
        if(instance.value(0) < mean){
            return 0.0;
        }
        return 1.0;
    }

    @Override
    //Predicts the class memberships for a given instance.
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] dist = new double[instance.numClasses()];
        double pred = classifyInstance(instance);
        dist[(int)pred] = 1 ;

        return dist;
    }

    public String toString(){
        return mean+"";
    }
}
