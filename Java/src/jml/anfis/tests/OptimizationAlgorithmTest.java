import jml.anfis.Anfis;
import jml.anfis.LSE_Optimization;
import jml.utils.FileOperations;
import jml.utils.MatrixOperations;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * Created by Jamal Hasanov on 6/18/2017.
 */
public class OptimizationAlgorithmTest {

    /**
     * Used to test LSE Optimization
     */
    @Test
    public void testLSE() {
        // Read INPUT and OUTPUT data from file:
        double[][] A = FileOperations.readData("unit_test_data/test_inputs_LSE.csv", ",");
        double[][] B = FileOperations.readData("unit_test_data/test_outputs_LSE.csv", ",");
        //Convert [P][1] to [1][P] and then keep only first row (converting 2D array into 1D)
        try {
            B = MatrixOperations.transpose(B);
        } catch (Exception e) {
            e.printStackTrace();
        }

        LSE_Optimization lse = new LSE_Optimization();
        double [] parameters = new double[A[0].length];
        // iterate over the training set - batch mode
        parameters = lse.findParameters(A, B[0]);

        // Check prameters (should be 1,2,3,4,5,6)
        System.out.print("Parameters: [");
        for (int i=0; i<parameters.length; i++) {
            System.out.print("" + parameters[i] + " ");
            assertTrue(Math.abs(parameters[i] - (i +1)) < 0.001);
        }
        System.out.println("] - passed");

    }
}
