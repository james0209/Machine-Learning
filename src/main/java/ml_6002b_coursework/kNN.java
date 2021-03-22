package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class kNN extends AbstractClassifier {
    private Instances instances;

    public int getK() {
        return k;
    }

    public void setK(int k) {
        this.k = k;
    }

    private int k = 5;
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        this.instances = instances;
    }

    public kNN( int k){
        this.k = k;
    }
    public double classifyInstance(Instance instance) {
        Instances instances = this.instances; //allows to delete without messing up internal instances
        double[][] kNN = new double[k][2];
        double previousDist = distance(instance, instances.get(0));
        int counter=0;
        double[][] temp = new double[2][2]; //swap in is 1, temp save is 2

        for (Instance dataInstance : instances) {
            double distance = distance(instance, dataInstance);
            if(counter < k ){ //adding it in for first k

                kNN[counter][0] = distance;
                kNN[counter][1] = dataInstance.value(instances.numAttributes() - 1);
            } else if (counter == k){
                kNN = sortkNN(kNN, k);
            } else if (counter > k && distance <= previousDist) {

                //Check through all the elements in kNN to see if closer.
                for(int i = 0; i < k; i++){

                    if(kNN[i][0]>distance){
                        //set up temp for first run
                        temp[0][0] = distance;
                        temp[0][1] = dataInstance.value(instances.numAttributes() - 1);

                        //run through to add in new closest distance
                        for (int j = i; j < kNN.length; j++){
                            temp[1][0] = kNN[i][0];
                            temp[1][1] = kNN[i][1];

                            kNN[i][0] = temp[0][0];
                            kNN[i][1] = temp[0][1];

                            temp[1][0] = temp[0][0];
                            temp[1][1] = temp[0][1];
                        }
                    }
                }
                //do the thing to check to put it in here
            }
            counter++;
        }
        //}
        return kNN[0][1];
    }

//    public double[][] insertDistance(double[][] kNN, double distance, int start){
//        //double[][] temp = kNN.clone();
//        double temp;
//        for (int i = start; i < kNN.length; i++){
//            temp = kNN[i][0];
//            kNN[i][0] = distance;
//            kNN[i][1] =
//        }
//
//    }

    public double[][] sortkNN(double[][] kNN, int k){
        boolean sorted = false;
        double temp;
        while(!sorted) {
            sorted = true;
            for (int i = 0; i < k - 1; i++) {
                if (kNN[i][0] > kNN[i+1][0]) {
                    //Sorts the distance values
                    temp = kNN[i][0];
                    kNN[i][0] = kNN[i+1][0];
                    kNN[i+1][0] = temp;

                    //Sorts the class types
                    temp = kNN[i][1];
                    kNN[i][1] = kNN[i+1][1];
                    kNN[i+1][1] = temp;

                    sorted = false;
                }
            }
        }
        return kNN;
    }
    public double[] distributionForInstance(Instance instance){
        double[] probability = new double[instance.attribute(instance.numAttributes()-1).numValues()];
        double result = ((int) classifyInstance(instance));

        probability[(int) result] = 1.0;

        return probability;
    }

    public double distance(Instance instance1, Instance instance2) {
        double difference;
        double totalDifference = 0;
        for (int i = 0; i < instance1.numAttributes() - 1; i++){
            difference = Math.pow((instance1.value(i) - instance2.value(i)),2);
            totalDifference = totalDifference + difference;
        }
        return  totalDifference;

    }
}
