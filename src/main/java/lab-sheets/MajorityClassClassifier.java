package lab_sheets;
// TRAINING CLASSIFIERS

import lab_sheets.HistogramClassifier;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class MajorityClassClassifier extends AbstractClassifier {
    int[] count;
    double[] classDistribution; //classDistirbuion is the proportion of the instances of each class value
    @Override
    public void buildClassifier(Instances data) throws Exception {
        count = new int[data.numClasses()];
        for(Instance ins:data){
            int c=(int)ins.classValue();
            count[c]++;
        }
        classDistribution= new double[data.numClasses()];
        for(int i=0;i<data.numClasses();i++)
            classDistribution[i]=count[i]/(double)data.numInstances();
    }
    @Override
    public double[] distributionForInstance(Instance ins){
        return classDistribution;
    }
    public String toString(){
        String str= "Class Distribution  = ";
        for(double d:classDistribution)
            str+=d+",";
        return str;
    }
    public static void main(String[] args) throws Exception {
        Instances all;
        String dataPath="C:\\Users\\Tony\\OneDrive - University of East Anglia\\Teaching\\2020-2021\\Machine " +
                "Learning\\Week 2 - Decision Trees\\Week 2 Live " +
                "Class\\tsml-master\\src\\main\\java\\experiments\\data\\uci\\iris\\";
        all=experiments.data.DatasetLoading.loadData(dataPath+"iris");
        MajorityClassClassifier mc= new MajorityClassClassifier();
        mc.buildClassifier(all);
        System.out.println(mc);
        int correct=0;
        for(Instance ins:all) {
            int pred = (int) mc.classifyInstance(ins);
            int actual = (int) ins.classValue();
            System.out.println(" Predicted  = "+pred);
            if (pred == actual)
                correct++;
//            double[] probs = mc.distributionForInstance(ins);
        }
        System.out.println(" Correct = "+correct+" Accuracy = "+(correct/(double)all.numInstances()));





/*

        //Build on all the iris data
        HistogramClassifier hc=new HistogramClassifier();
        hc.buildClassifier(all);
        System.out.println("MODEL = "+hc.toString());
        int correct=0;
        for(Instance ins:all){
            int pred=(int)hc.classifyInstance(ins);
            int actual = (int)ins.classValue();
            System.out.println(" Actual = "+actual+" Predicted ="+pred);
            if(pred==actual)
                correct++;
        }
        System.out.println(" num correct = "+correct+" accuracy  = "+ (correct/(double)all.numInstances()));
        Instances[] split = InstanceTools.resampleInstances(all,0,0.5);
*/

    }





}
