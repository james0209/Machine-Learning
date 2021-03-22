package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

    public class OneNN extends AbstractClassifier {
        private Instances instances;
        @Override
        public void buildClassifier(Instances instances) throws Exception {
            this.instances = instances;
        }

        public double classifyInstance(Instance instance) {
            double previousDist = distance(instance, instances.get(0));
            double classType = 0;
            for (Instance dataInstance : instances) {
                double distance = distance(instance, dataInstance);
                if (distance <= previousDist) {
                    previousDist = distance;
                    classType = dataInstance.value(instances.numAttributes()-1);
                }
            }
            return classType;
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
