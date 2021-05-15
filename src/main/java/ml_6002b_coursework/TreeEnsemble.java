package ml_6002b_coursework;

import evaluation.tuning.ParameterSpace;
import tsml.classifiers.Tuneable;
import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.filters.unsupervised.attribute.RandomSubset;

import java.util.*;
import java.util.stream.IntStream;

import static ml_6002b_coursework.WekaTools.loadClassificationData;
import static ml_6002b_coursework.WekaTools.splitData;
import static utilities.InstanceTools.resampleInstances;
import static utilities.Utilities.argMax;

public class TreeEnsemble extends AbstractClassifier implements Tuneable {
    private int seed = 0;
    private int ensembleSize = 50;
    private double proportion = 0.5;
    private boolean averaging = false;
    private ID3Coursework baseClassifier = new ID3Coursework();
    private final LinkedHashMap<ID3Coursework, RandomSubset> attributesUsed = new LinkedHashMap<>();

    public int getSeed() {
        return seed;
    }

    public void setSeed(int seed) {
        this.seed = seed;
    }

    public int getensembleSize() {
        return ensembleSize;
    }

    public void setensembleSize(int ensembleSize) {
        this.ensembleSize = ensembleSize;
    }

    public double getProportion() {
        return proportion;
    }

    public void setProportion(double proportion) {
        this.proportion = proportion;
    }

    public void setBaseClassifier(ID3Coursework baseClassifier) {
        this.baseClassifier = baseClassifier;
    }

    public boolean isAveraging() {
        return averaging;
    }

    public void setAveraging(boolean averaging) {
        this.averaging = averaging;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String t = Utils.getOption('t', options);
        String p = Utils.getOption('p', options);
        String a = Utils.getOption('a', options);
        if (!t.equals(""))
            ensembleSize = Integer.parseInt(t);
        if (!p.equals(""))
            proportion = Double.parseDouble(p);
        if (!a.equals(""))
            averaging = Boolean.parseBoolean(a);
        baseClassifier.setOptions(options);
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        System.out.println("building Tree ensemble \n" +
                ensembleSize + " trees\n" +
                proportion + " attributes\n" +
                (averaging? "averaging\n":""));

        Random random= new Random(seed);


        for (int i = 0; i < ensembleSize; i++) {
            RandomSubset attSubset = new RandomSubset();
            attSubset.setNumAttributes(proportion);
            //attSubset.setSeed(seed + i);
            attSubset.setInputFormat(data);
            Instances inst = attSubset.process(data);

            ID3Coursework id3 = new ID3Coursework();
            id3 = baseClassifier;

            String[] options = new String[1];

            int temp = random.nextInt(4);
            switch (temp){
                case 0:
                    options[0] = "-I";
                    break;
                case 1:
                    options[0] = "-G";
                    break;
                case 2:
                    options[0] = "-C";
                    break;
                case 3:
                    options[0] = "-Y";
                    break;
            }

            id3.setOptions(options);
            id3.buildClassifier(inst);
            attributesUsed.put(id3, attSubset);
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] distribution = distributionForInstance(instance);
        //int[] temp = argMax(distribution);
        return argMax(distribution, new Random());

    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

        double[] distribution = new double[instance.numClasses()];

        attributesUsed.forEach((key, value) -> {
            ID3Coursework id3 = key;
            try {
                value.setInputFormat(instance.dataset());
                value.input(instance);
                Instance inst = value.output();

                if (averaging){
                    double[] dist = id3.distributionForInstance(inst);
                    IntStream.range(0, dist.length).forEach(i -> distribution[i] += dist[i]);
                }
                else{
                    distribution[(int) id3.classifyInstance(inst)]++;
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
        });

        for (int i = 0; i < distribution.length; i++) {
            distribution[i] /= ensembleSize;
        }

        return distribution;
    }

    @Override
    public Capabilities getCapabilities() {
        return baseClassifier.getCapabilities();
    }

    public static void main(String[] args) throws Exception {
        Instances trainingData = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/optdigits.arff");
        Instances[] trainTest = splitData(trainingData, 0.7);
        Instances optdigitsTrain = trainTest[0];
        Instances optdigitsTest = trainTest[1];
        TreeEnsemble treeEnsemble = new TreeEnsemble();

        try
        {
            treeEnsemble.setAveraging(true);
            treeEnsemble.buildClassifier(optdigitsTrain);
            System.out.println("optdigits test accuracy: " + WekaTools.accuracy(treeEnsemble, optdigitsTest));
            System.out.println("probability estimates for the first five test cases: ");
            for (int i = 0; i < 5; i++)
                System.out.println("classify instance = " + treeEnsemble.classifyInstance(trainingData.get(i)));
        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    @Override
    public ParameterSpace getDefaultParameterSearchSpace() {
        ParameterSpace ps = new ParameterSpace();
/*        String[] ensembleSize={"50","100","300","500"};
        ps.addParameter("t", ensembleSize);
        String[] proportion={"0.5","0.75","1"};
        ps.addParameter("p", proportion);*/
        String[] averaging={"true","false"};
        ps.addParameter("a", averaging);
        ps.parameterLists.putAll(baseClassifier.getDefaultParameterSearchSpace().parameterLists);
        return ps;
    }
}
