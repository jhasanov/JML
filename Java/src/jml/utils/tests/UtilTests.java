package jml.utils.tests;

import jml.utils.FileOperations;
import jml.utils.MatrixOperations;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * Created by itjamal on 6/13/2017.
 */
public class UtilTests {

    @Test
    public void testFileOperations() {
        double [][] fileData = FileOperations.readData("D:\\Aeroimages\\baku\\Jamal\\Codes\\Matlab\\ANFIS\\HS_inputs.csv",",",-1,-1,false);

        System.out.print("\nTesting input data...");
        assertTrue(fileData.length == 6518);
        assertTrue(fileData[0].length == 2);
        System.out.println(" passed");

        System.out.print("Testing output data...");
        fileData = FileOperations.readData("D:\\Aeroimages\\baku\\Jamal\\Codes\\Matlab\\ANFIS\\HS_outputs.csv",",",-1,-1,false);

        assertTrue(fileData.length == 6518);
        assertTrue(fileData[0].length == 1);
        System.out.println(" passed");
    }
}
