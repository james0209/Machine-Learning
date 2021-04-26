package ml_6002b_coursework;

public class AttributeMeasures {
    private static final double log2 = Math.log(2);

    // Rows represent different values of the attribute being assessed
    // Columns represent the class counts

    public static double measureInformationGain(double[][] array){
        double returnValue = 0, rowSum, total = 0;
        int numRows = array.length;
        int numCols = array[0].length;

        for (double[] doubles : array) {
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

    public static double measureGini(double[][] array){
        try{
            double returnValue = 0, rowSum = 0, total = 0, weighted = 0;
            int numRows = array.length;
            int numCols = array[0].length;
            double[] values = new double[numCols];

            for (double[] doubles : array) {
                for (int j = 0; j < numCols; j++) {
                    total += doubles[j];
                }
            }


            for (double[] doubles : array) {
                returnValue = 0;
                rowSum = 0;
                for (int j = 0; j < numCols; j++) {
                    rowSum += doubles[j];
                    values[j] = doubles[j];
                }

                for (double x : values){
                    //System.out.println(x + " / " + rowSum);
                    returnValue += Math.pow((x/rowSum),2);
                    //System.out.println("Return value " + returnValue);
                }


                returnValue = (1 - returnValue);
                //System.out.println("Calculated impurity for node  " + returnValue);
                //System.out.println("Calculation  " + returnValue + " * " + "( " + rowSum + " / " + total);
                weighted+= (returnValue*(rowSum/total));
                //System.out.println("Current weighted = " + weighted);
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

    public static double measureChiSquared(double[][] array){
        double chiSquareValue = 0.0;
        try{
            int row, col;
            double total = 0;
            double[] rowSum, colSum;

            int numRows = array.length;
            int numCols = array[0].length;

            rowSum = new double [numRows];
            colSum  = new double [numCols];

            for (row = 0; row < numRows; row++) {
                for (col = 0; col < numCols; col++) {
                    rowSum[row] += array[row][col];
                    colSum [col] += array[row][col];
                    total += array[row][col];
                }
            }

            // Ensure that data is at least a 2x2 table matrix
            // DF = degrees of freedom
            int df = (numRows - 1)*(numCols - 1);
            if (df <= 0) {
                return 0.0;
            }
            if ( numRows != numCols ){
                return 0.0;
            }

            double expected;

            for (row = 0; row < numRows; row++) {
                for (col = 0; col < numCols; col++) {
                    expected = (colSum[col] * rowSum[row]) / total;
                    double difference = Math.abs(array[row][col] - expected);
                    chiSquareValue += (difference * difference / expected);
                }
            }
            return chiSquareValue;
        }
        catch (Exception e){
            e.printStackTrace();
        }
        return chiSquareValue;
    }

    public static double measureChiSquaredYates(double[][] array){
        double chiSquareYatesValue = 0.0;
        try{
            int row, col;
            double total = 0;
            double[] rowSum, colSum;

            int numRows = array.length;
            int numCols = array[0].length;

            rowSum = new double [numRows];
            colSum  = new double [numCols];

            for (row = 0; row < numRows; row++) {
                for (col = 0; col < numCols; col++) {
                    rowSum[row] += array[row][col];
                    colSum [col] += array[row][col];
                    total += array[row][col];
                }
            }

            // Ensure that data is at least a 2x2 table matrix
            // DF = degrees of freedom
            int df = (numRows - 1)*(numCols - 1);
            if (df <= 0) {
                return 0.0;
            }

            double expected;

            for (row = 0; row < numRows; row++) {
                for (col = 0; col < numCols; col++) {
                    expected = (colSum[col] * rowSum[row]) / total;
                    double difference = Math.abs(array[row][col] - expected);
                    difference -= 0.5;
                    chiSquareYatesValue += (difference * difference / expected);
                }
            }
            return chiSquareYatesValue;
        }
        catch (Exception e){
            e.printStackTrace();
        }
        return chiSquareYatesValue;
    }



    public static void main(String[] args) throws Exception {
        double[][] array = {{3, 2}, {3, 4}};
        System.out.println("measure information gain for headache splitting diagnosis = " + measureInformationGain(array));
        System.out.println("measure gini for headache splitting diagnosis = " + measureGini(array));
        System.out.println("measure chi-squared for headache splitting diagnosis = " + measureChiSquared(array));
        System.out.println("measure chi-squared with yates correction for headache splitting diagnosis = "
                + measureChiSquaredYates(array));

    }
}
