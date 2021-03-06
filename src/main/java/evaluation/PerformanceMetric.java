/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */

package evaluation;

import evaluation.storage.ClassifierResults;
import java.util.ArrayList;
import java.util.function.Function;

/**
 * This is essentially a placeholder class, and may be expanded into a full package
 * with this being an abstract base class etc. 
 * 
 * For now, this is a container class for metrics and meta info about them. It contains
     - The name of this metric for printouts
     - A function to get this metric's score fro ma classifier results object (in future, perhaps calculate them here instead, etc)
     - A flag for whether this metric wants to be maximise or minimised
     - A flag to _suggest_ how this metric should be summarised/averaged 
                - for now, mean vs median for e.g accs vs timings. For timings we would want to 
                  use median instead of mean to reduce effect of outliers
                - in future, probably just define a comparator
     - A descriptor for use in images when comparing classifiers with this metric, e.g better/worse/slower
     - Maybe more in the future     
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class PerformanceMetric {

    public String name;
    public Function<ClassifierResults, Double> getter;
    public boolean takeMean;
    public boolean maximise;
    
    /**
     * currently only used for the pairwise scatter diagrams in the pipeline, 
     * this refers to the descriptor for comparing the scores of a metric between 
     * classifiers
     *
     * If the raw value of a is HIGHER than b, then a is {better,worse,slower,faster,etc.} than b
     */
    public String comparisonDescriptor;
    
    public PerformanceMetric(String metricName, Function<ClassifierResults, Double> getScore, boolean takeMean, boolean maximised, String comparisonDescriptor) {
        this.name = metricName;
        this.getter = getScore;
        this.takeMean = takeMean;
        this.maximise = maximised;
        this.comparisonDescriptor = comparisonDescriptor;
    }
    
    public double getScore(ClassifierResults res) {
        return getter.apply(res);
    }
    
    public String toString() { 
        return name;
    }
    
    private static final boolean min = false, max = true;
    private static final boolean median = false, mean = true;
    private static final String better = "better", worse = "worse", slower = "slower", faster = "faster";
    
    public static PerformanceMetric acc             = new PerformanceMetric("ACC", ClassifierResults.GETTER_Accuracy,                    mean, max,   better);
    public static PerformanceMetric balacc          = new PerformanceMetric("BALACC", ClassifierResults.GETTER_BalancedAccuracy,         mean, max,   better);
    public static PerformanceMetric AUROC           = new PerformanceMetric("AUROC", ClassifierResults.GETTER_AUROC,                     mean, max,   better);
    public static PerformanceMetric NLL             = new PerformanceMetric("NLL", ClassifierResults.GETTER_NLL,                         mean, min,   worse);
    public static PerformanceMetric F1              = new PerformanceMetric("F1", ClassifierResults.GETTER_F1,                           mean, max,   better);
    public static PerformanceMetric MCC             = new PerformanceMetric("MCC", ClassifierResults.GETTER_MCC,                         mean, max,   better);
    public static PerformanceMetric precision       = new PerformanceMetric("Prec", ClassifierResults.GETTER_Precision,                  mean, max,   better);
    public static PerformanceMetric recall          = new PerformanceMetric("Recall", ClassifierResults.GETTER_Recall,                   mean, max,   better);
    public static PerformanceMetric sensitivity     = new PerformanceMetric("Sens", ClassifierResults.GETTER_Sensitivity,                mean, max,   better);
    public static PerformanceMetric specificity     = new PerformanceMetric("Spec", ClassifierResults.GETTER_Specificity,                mean, max,   better);

    public static PerformanceMetric buildTime       = new PerformanceMetric("TrainTimes", ClassifierResults.GETTER_buildTimeDoubleMillis,         median, min, slower);
    public static PerformanceMetric totalTestTime   = new PerformanceMetric("TestTimes", ClassifierResults.GETTER_totalTestTimeDoubleMillis,      median, min, slower);
    public static PerformanceMetric avgTestPredTime = new PerformanceMetric("AvgPredTimes", ClassifierResults.GETTER_avgTestPredTimeDoubleMillis, median, min, slower);
    public static PerformanceMetric fromScratchEstimateTime = new PerformanceMetric("FromScratchEstimateTimes", ClassifierResults.GETTER_fromScratchEstimateTimeDoubleMillis, median, min, slower);
    public static PerformanceMetric totalBuildPlusEstimateTime = new PerformanceMetric("TotalBuildPlusEstimateTimes", ClassifierResults.GETTER_totalBuildPlusEstimateTimeDoubleMillis, median, min, slower);
    public static PerformanceMetric additionalTimeForEstimate = new PerformanceMetric("AdditionalTimesForEstimates", ClassifierResults.GETTER_additionalTimeForEstimateDoubleMillis, median, min, slower);

    public static PerformanceMetric buildTimeBenchmarked = new PerformanceMetric("TrainTimesBenchmarked", ClassifierResults.GETTER_buildTimeDoubleMillisBenchmarked,         median, min, slower);
    public static PerformanceMetric totalTestTimeBenchmarked = new PerformanceMetric("TestTimesBenchmarked", ClassifierResults.GETTER_totalTestTimeDoubleMillisBenchmarked,      median, min, slower);
    public static PerformanceMetric avgTestPredTimeBenchmarked = new PerformanceMetric("AvgPredTimesBenchmarked", ClassifierResults.GETTER_avgTestPredTimeDoubleMillisBenchmarked, median, min, slower);
    public static PerformanceMetric fromScratchEstimateTimeBenchmarked = new PerformanceMetric("FromScratchEstimateTimesBenchmarked", ClassifierResults.GETTER_fromScratchEstimateTimeDoubleMillisBenchmarked, median, min, slower);
    public static PerformanceMetric totalBuildPlusEstimateTimeBenchmarked = new PerformanceMetric("TotalBuildPlusEstimateTimesBenchmarked", ClassifierResults.GETTER_totalBuildPlusEstimateTimeDoubleMillisBenchmarked, median, min, slower);
    public static PerformanceMetric additionalTimeForEstimateBenchmarked = new PerformanceMetric("AdditionalTimesForEstimatesBenchmarked", ClassifierResults.GETTER_additionalTimeForEstimateDoubleMillisBenchmarked, median, min, slower);

    public static PerformanceMetric benchmarkTime = new PerformanceMetric("BenchmarkTimes", ClassifierResults.GETTER_benchmarkTime, median, min, slower);
    public static PerformanceMetric memory          = new PerformanceMetric("MaxMemory", ClassifierResults.GETTER_MemoryMB,                median, min,   worse);

    public static PerformanceMetric earliness       = new PerformanceMetric("Earliness", ClassifierResults.GETTER_Earliness,             mean, min,   worse);
    public static PerformanceMetric harmonicMean    = new PerformanceMetric("HarmonicMean", ClassifierResults.GETTER_HarmonicMean,       mean, max,   better);

    
    public static ArrayList<PerformanceMetric> getAccuracyStatistic() { 
        ArrayList<PerformanceMetric> stats = new ArrayList<>();
        stats.add(acc);
        return stats;
    }
    
    public static ArrayList<PerformanceMetric> getDefaultStatistics() { 
        ArrayList<PerformanceMetric> stats = new ArrayList<>();
        stats.add(acc);
        stats.add(balacc);
        stats.add(AUROC);
        stats.add(NLL);
        return stats;
    }
        
    public static ArrayList<PerformanceMetric> getAllStatistics() { 
        ArrayList<PerformanceMetric> stats = new ArrayList<>();
        stats.add(acc);
        stats.add(balacc);
        stats.add(AUROC);
        stats.add(NLL);
        stats.add(F1);
        stats.add(MCC);
        stats.add(precision);
        stats.add(recall);
        stats.add(sensitivity);
        stats.add(specificity);

        //stats.add(memory);

        //stats.add(earliness);
        //stats.add(harmonicMean);

        return stats;
    }
}
