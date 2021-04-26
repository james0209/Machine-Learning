package topic1_getting_started;

import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.meta.LogitBoost;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Sketch solution to lab sheet 1 question 6:
 *6) Making Your Own Classifier
 * In week 1 we describe a simple histogram classifier. This works for a single attribute, which it converts into discrete intervals.
 * 1.	Create a new class called HistogramClassifier that extends the Classifier class. Assume a default number of
 * 10 bins and that the attribute to be used is in position 0. Write methods so the user can alter these defaults.
 * 2.	Implement buildClassifier. This method should construct the histograms based on the training data. You should:
 *      a.	Allocate the memory for the histogram for each class (you can find out the number of classes from the
 *      training data by calling numClasses()).
 *      b.	Set fixed interval values for the bins from the minimum up to the maximum for the attribute
 *      (you need to figure out how to calculate the min and max value of an attribute)
 *      c.	Count the occurrences in each field (use a nested if statement or if you are very clever use the
 *      divide and modulus operators).
 * 3.	Implement distributionForInstance. This method should first find the correct interval for the data passed as an
 * argument, then work out the relative frequencies of the histograms, then set the probability for each class value.
 * 4.	Implement classify instance. This should simply call distributionForInstance and pick the maximum.
 * 5.	Enter the following training data into an ARFF file, work out the histograms by hand, then test your classifier.
 * Experiment with altering the parameters of the model (number of bins and range of values).
 *
 */

public class HistogramClassifier implements Classifier {
    int numBins=10;
    int[][] counts;
    int attribute=0;
    double min, max, interval;
    @Override
    public void buildClassifier(Instances data) throws Exception {
        int numClasses=data.numClasses();
        counts=new int[numClasses][numBins];
//Find min and max of attribute
        min=data.attributeStats(attribute).numericStats.min;
        max=data.attributeStats(attribute).numericStats.max;
        double range =max-min;
        interval=range/numBins; //Will this capture all
        System.out.println(" min= "+min);
        System.out.println(" max= "+max);
        System.out.println(" interval= "+interval);

        for(Instance ins:data){
            double value=ins.value(attribute);
            int classVal=(int)ins.classValue();
            //Find bin from value: slow way of doing it, could use integer division if we are clever
            double x=min;
            int c=0;
            while(x<value){
                x+=interval;
                c++;
            }
            if(x>=max)
                c=numBins-1;
            counts[classVal][c]++;
        }
    }
    private static int maxIndex( int[] x){
        int index=0;
        for(int i=1;i<x.length;i++){
            if(x[i]>x[index])
                index=i;
        }
        return index;
    }
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double value=instance.value(attribute);
//Find which bin its in
        //Find bin from value
        double x=min;
        int c=0;
        while(x<value){
            x+=interval;
            c++;
        }
        if(x>=max)
            c=numBins-1;
        //Find max count: Maybe more efficient to transpose counts completely?
        int[] ct=new int[instance.numClasses()];
        for(int i=0;i<counts.length;i++)
            ct[i]=counts[i][c];
        return maxIndex(ct);

    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] dist = new double[instance.numClasses()];
        double value=instance.value(attribute);
//Find which bin its in
        //Find bin from value
        double x=min;
        int c=0;
        while(x<value){
            x+=interval;
            c++;
        }
        if(x>=max)
            c=numBins-1;
        //Find max count: Maybe more efficient to transpose counts completely?
        double sum=0;
        for(int i=0;i<counts.length;i++) {
            dist[i] = counts[i][c];
            sum += dist[i];
        }
        for(int i=0;i<counts.length;i++)
            dist[i]/=sum;
        return dist;
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }



    @Override
    public String toString(){
        String res= "interval = "+interval+"\n";
        res+= "Histogram::\n";
        for(int[] hist:counts) {
            for (int c : hist)
                res += c + ",";
            res += "\n";
        }
        return res;
    }

    public static void main(String[] args) throws Exception {
        Instances all;
        String dataPath="C:\\Users\\Tony\\OneDrive - University of East Anglia\\Teaching\\2020-2021\\Machine " +
                "Learning\\Week 2 - Decision Trees\\Week 2 Live " +
                "Class\\tsml-master\\src\\main\\java\\experiments\\data\\uci\\iris\\";
        all=experiments.data.DatasetLoading.loadData(dataPath+"iris");
        //Build on all the iris data
        HistogramClassifier hc=new HistogramClassifier();
        hc.buildClassifier(all);
        System.out.println("MODEL = "+hc.toString());


        int correct=0;
        for(Instance ins:all){
            int pred=(int)hc.classifyInstance(ins);
            int actual = (int)ins.classValue();
//            System.out.println(" Actual = "+actual+" Predicted ="+pred);
            if(pred==actual)
                correct++;
        }
        System.out.println(" num correct = "+correct+" accuracy  = "+ (correct/(double)all.numInstances()));
        Instances[] split = InstanceTools.resampleInstances(all,0,0.5);
        hc.buildClassifier(split[0]);
        System.out.println("MODEL = "+hc.toString());

        correct=0;
        for(Instance ins:split[1]){
            int pred=(int)hc.classifyInstance(ins);
            int actual = (int)ins.classValue();
//            System.out.println(" Actual = "+actual+" Predicted ="+pred);
            if(pred==actual)
                correct++;
        }
        System.out.println(" num correct = "+correct+" accuracy  = "+ (correct/(double)split[1].numInstances()));




    }
}














