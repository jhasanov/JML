package jml.utils;

/**
 * Created by itjamal on 1/24/2017.
 */
public class MatrixOperations {

    public static double[][] transpose(double [][] matrixX) throws Exception{
        int cols = matrixX.length;
        int rows = matrixX[0].length;

        // check for valid length
        if ((cols * rows) <= 0)
            throw new Exception("Invalid size");

        double [][] result = new double[rows][cols];

        return result;
    }

    // Simple multiplication method
    public static double[][] multiplySimple(double [][] matrixA, double [][] matrixB) throws Exception{
        // perform checks
        int rowsA = matrixA.length;
        int colsA = matrixA[0].length;
        int rowsB = matrixB.length;
        int colsB = matrixB[0].length;

        if (colsA != rowsB) {
            throw new Exception("Multiplied matrix column and row count doesn't match!");
        }

        double [][] product = new double[rowsA][colsB];

        for (int i=0; i< rowsA; i++)
            for (int j=0; j<colsB; j++)
                for (int k = 0; k < colsA; k++)
                    product[i][j] += matrixA[i][k] * matrixB[k][j];

        return product;
    }

    // addidion and subtraction is done by the same function
    // if SIGN is -1, then it's subtraction, if 1 , then addition
    public static double[][] sum(double[][] matrixA, double [][] matrixB, int sign) throws Exception{
        // perform checks
        int rowsA = matrixA.length;
        int colsA = matrixA[0].length;
        int rowsB = matrixB.length;
        int colsB = matrixB[0].length;

        if ((colsA != colsA) || (rowsA != rowsB)) {
            throw new Exception("Matrix size doesn't match!");
        }

        double [][] result = matrixA;

        for (int i=0; i<rowsA; i++)
            for (int j=0; j<colsA; j++)
                result[i][j] += matrixB[i][j]*sign;

        return result;
    }

    // Addition is called with Sign=1
    public static double[][] add(double[][] matrixA, double [][] matrixB) throws Exception{
        return sum(matrixA,matrixB,1);
    }

    // Subtraction is called with Sign=1
    public static double[][] subtract(double[][] matrixA, double [][] matrixB) throws Exception{
        return sum(matrixA,matrixB,-1);
    }

    public static double[][] multiplyByNumber(double[][] matrixX,double val) throws Exception {
        for (int i=0; i<matrixX.length; i++)
            for (int j=0; j<matrixX[0].length; j++)
                matrixX[i][j] = matrixX[i][j] * val;

        return matrixX;
    }

    public static double[][] divideByNumber(double[][] matrixX,double val) throws Exception {
        if (val == 0)
            throw new NullPointerException("Division by Zero!");

        for (int i=0; i<matrixX.length; i++)
            for (int j=0; j<matrixX[0].length; j++)
                matrixX[i][j] = matrixX[i][j] / val;

        return matrixX;
    }

}
