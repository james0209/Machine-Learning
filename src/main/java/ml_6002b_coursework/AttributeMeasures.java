package ml_6002b_coursework;

import static java.lang.Double.max;

public class AttributeMeasures {
    private static final double log2 = Math.log(2);
    private static double sumDoubles(double[] x){
        double sum = 0;
        for (int i = 0;i < x.length;i++){
            sum += x[i];
        }
        return sum;
    }

    private static int sumInts(int[] x){
        int sum = 0;
        for (int i = 0;i < x.length;i++){
            sum += x[i];
        }
        return sum;
    }

    // Rows represent different values of the attribute being assessed
    // Columns represent the class counts

    public static double measureInformationGain(int[][] array){
        if (array.length == 0 || array[0].length == 0)
            return 0;

        double returnValue = 0, rowSum, total = 0;
        int numRows = array.length;
        int numCols = array[0].length;

        for (int[] doubles : array) {
            rowSum = 0;
            for (int j = 0; j < numCols; j++) {
                returnValue = returnValue + (doubles[j] * Math.log(doubles[j]));
                rowSum += doubles[j];
            }
            returnValue = returnValue - (rowSum * Math.log(rowSum));
            total += rowSum;
        }
        try{
            return 1-(-returnValue / (total * log2));
        }
        catch (Exception e){
            e.printStackTrace();
            return 0.0;
        }

    }

    public static double measureGini(int[][] array){
        if (array.length == 0 || array[0].length == 0)
            return 0;

        try{
            double returnValue = 0, rowSum = 0, total = 0, weighted = 0;
            int numRows = array.length;
            int numCols = array[0].length;
            double[] values = new double[numCols];

            for (int[] doubles : array) {
                for (int j = 0; j < numCols; j++) {
                    total += doubles[j];
                }
            }


            for (int[] doubles : array) {
                returnValue = 0;
                rowSum = 0;
                for (int j = 0; j < numCols; j++) {
                    rowSum += doubles[j];
                    values[j] = doubles[j];
                }

                for (double x : values){
                    returnValue += Math.pow((x/rowSum),2);
                }


                returnValue = (1 - returnValue);
                weighted+= (returnValue*(rowSum/total));
            }

            if (total == 0) {
                return 0;
            }

            return weighted;

        }
        catch (Exception e){
            e.printStackTrace();
            return 0.0;
        }
    }

    private static double chiSquared(int[][] split, boolean yates){
        double chi = 0, count = 0;

        int attributeValues = split.length;
        int classes = split[0].length;

        double[] attTotals = new double [attributeValues];
        double[] classTotals = new double [classes];

        for (int row = 0; row < attributeValues; row++) {
            for (int col = 0; col < classes; col++) {
                double classCount = split[row][col];
                attTotals[row] += classCount;
                classTotals[col] += classCount;
                count += classCount;
            }
        }

        for (int row = 0; row < attributeValues; row++) {
            if (attTotals[row] > 0) {
                for (int col = 0; col < classes; col++) {
                    if (classTotals[col] > 0) {
                        double expected = attTotals[row] * (classTotals[col] / count);
                        double diff = Math.abs(split[row][col] - expected);
                        if(yates) {
                            diff = max(diff - 0.5, 0);
                        }
                        chi += diff * diff / expected;
                    }
                }
            }
        }
        return chi;
    }
    public static double measureChiSquared(int[][] array) {
        return chiSquared(array, false);
    }

    public static double measureChiSquaredYates(int[][] array) {
        return chiSquared(array, true);
    }

    public static void main(String[] args) throws Exception {
        int[][] array = {{3, 2}, {3, 4}};
        System.out.println("measure information gain for headache splitting diagnosis = " + measureInformationGain(array));
        System.out.println("measure gini for headache splitting diagnosis = " + measureGini(array));
        System.out.println("measure chi-squared for headache splitting diagnosis = " + measureChiSquared(array));
        System.out.println("measure chi-squared with yates correction for headache splitting diagnosis = "
                + measureChiSquaredYates(array));

    }
}
