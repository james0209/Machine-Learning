package ml_6002b_coursework;

import java.util.Arrays;

public class AttributeMeasures {

    public static double measureInformationGain(int[][] array){
        //H(X) is the entropy at root
        //Information Gain = entropy(Parent) - [average entropy(children)]
        //                  ==
        try{
            return 0.0;
        }
        catch (Exception e){
            e.printStackTrace();
        }
        return 0.0;
    }

    public static double measureGini(int[][] array){
        try{
            return 0.0;
        }
        catch (Exception e){
            e.printStackTrace();
        }
        return 0.0;
    }

    public static double measureChiSquared(int[][] array){
        try{
            int degFreedom = (array.length - 1) * (array[0].length - 1);
            return 0.0;
        }
        catch (Exception e){
            e.printStackTrace();
        }
        return 0.0;
    }

    public static double measureChiSquaredYates(int[][] array){
        try{
            return 0.0;
        }
        catch (Exception e){
            e.printStackTrace();
        }
        return 0.0;
    }

    public String toString() {
        return "measure " + "insert here " + "for headache splitting diagnosis = " + "insert here";
    }

    public static void main(String[] args) throws Exception {
        int DEPT_MATH = 1;
        int DEPT_HISTORY = 2;
        int DEPT_CS = 3;

        int YES = 1;
        int NO = 2;

        String [] deptAttributeString = {"Math", "History", "CS"};
        int [] deptAttribute = {DEPT_MATH, DEPT_HISTORY, DEPT_CS};
        int [] Y_values= {YES, NO};
        //System.out.println(Arrays.deepToString(arr).replace("], ", "]\n"));
        //measureInformationGain(arr);
    }
}
