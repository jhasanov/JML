package jml.utils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.StringTokenizer;

/**
 * Created by itjamal on 6/13/2017.
 */
public class FileOperations {

    public static double [][] readData(String inputDataFile, String delimeter) {
        double [][] inputs = new double[0][0];
        int sampleCnt, varCnt;

        try {
            InputStream fis = new FileInputStream(inputDataFile);
            InputStreamReader isr = new InputStreamReader(fis);
            BufferedReader br = new BufferedReader(isr);

            ArrayList list = new ArrayList<>();
            String line = "";
            StringTokenizer st;
            double [] elems = new double[0]; // formally initializing it to use after "while" loop

            while ((line = br.readLine())!= null) {
                st = new StringTokenizer(line,delimeter);
                elems = new double[st.countTokens()];
                int i = 0;
                while (st.hasMoreTokens()) {
                    elems[i++] = Double.parseDouble(st.nextToken());
                }
                list.add(elems);
            }
            inputs = new double[list.size()][elems.length];

            for (int i=0; i<list.size(); i++) {
                inputs[i] = (double [])list.get(i);
            }
        }
        catch (Exception ex) {
            System.out.println("Anfis.readInputs(): "+ex);
        }

        return inputs;
    }

}
