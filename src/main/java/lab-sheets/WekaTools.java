
package lab_sheets;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import java.io.FileReader;
import java.util.Arrays;

public class WekaTools {
    static Instances data;

    public static Instances loadClassificationData(String fullPath){
        try{
            FileReader reader = new FileReader(fullPath);
            data = new Instances(reader);
            return data;
        }
        catch(Exception e){
            System.out.println("Exception caught: "+e);
        }
        int numInstances = data.numInstances();
        int numAttributes = data.numAttributes();
        data.setClassIndex(numAttributes-1);

        return data;
    }

    public static Instances[] splitData(Instances all, double proportion){
        Instances[] split = new Instances[2];

        split[0]=new Instances(all); // Train
        split[1]=new Instances(all,0); // Test

        int splitIndex = (int)Math.round(proportion * (double)all.numInstances());

        //TODO: Implement method here to randomise data before splitting
        //in case data is pre-ordered by class value etc.

        for(int i = 0; i < splitIndex; i++)
        {
            Instance instanceToMove = split[0].remove(0);
            split[1].add(instanceToMove);
        }

        return split;
    }

    public static double accuracy(Classifier c, Instances test) throws Exception {
        int numInstances = data.numInstances();

        int correct=0;

        for(Instance instance : test){
            int pred=(int)c.classifyInstance(instance);
            int actual = (int)instance.classValue();

            if(pred==actual)
                correct++;
        }
        //The accuracy of the classifier is then the number correct divided by the number of instances
        double accuracy = (correct/(double)data.numInstances());
        return accuracy;
    }

    public static double[] classDistribution(Instances data){
        double[] classDistribution = new double[data.numClasses()];


        for(int i = 0; i < data.numInstances(); i++)
        {
            classDistribution[(int)data.get(i).value(data.numAttributes() - 1)]++;
        }

        return classDistribution;
    }

}

