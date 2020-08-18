package jml.utils;

import com.sun.xml.txw2.output.IndentingXMLStreamWriter;
import jml.anfis.*;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamWriter;
import java.io.*;
import java.util.ArrayList;
import java.util.StringTokenizer;

/**
 * Created by Jamal Hasanov on 6/13/2017.
 */
public class FileOperations {

    public static void appendToFile(String fileName, String text) {
        try (FileWriter fr = new FileWriter(fileName,true); ) {
            fr.write(text);
        }
        catch (IOException iex) {
            System.out.println("FileOperations.appendToFile : "+iex);
        }
    }

    public static double [][] readData(String inputDataFile, String delimeter,int startColInclusive, int endColExclusive,boolean bIgnoreFirstLine) {
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

            if (bIgnoreFirstLine) {
                br.readLine();
            }

            while ((line = br.readLine())!= null) {
                st = new StringTokenizer(line,delimeter);
                if ((startColInclusive == -1) && (endColExclusive == -1))
                    elems = new double[st.countTokens()];
                else {
                    int statCol = (startColInclusive == -1) ? 0 : startColInclusive;
                    int endCol = (endColExclusive == -1) ? st.countTokens() : endColExclusive;
                    elems = new double[endCol-statCol];
                }
                int i = 0;
                int colNum = 0;
                while (st.hasMoreTokens()) {
                    if (((startColInclusive == -1) || (colNum >= startColInclusive)) &&
                            ((endColExclusive == -1) || (colNum < endColExclusive)) ) {
                        elems[i++] = Double.parseDouble(st.nextToken());
                    }
                    else {
                        st.nextToken();
                    }
                    colNum++;
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

    /**
     * Saves ANFIS structure and parameters in a file
     *
     * @param filename XML file to store the ANFIS data
     */
    public static void saveAnfisToFile(Anfis anfis, String filename) {
        try {
            System.out.println("Saving ANFIS to file: " + filename);
            OutputStream outputStream = new FileOutputStream(new File(filename));
            XMLOutputFactory xmlOF = XMLOutputFactory.newInstance();
            // This line of code makes XML structure formatted
            XMLStreamWriter xmlWrite = new IndentingXMLStreamWriter(xmlOF.createXMLStreamWriter(outputStream));

            xmlWrite.writeStartDocument();
            xmlWrite.writeStartElement("anfis");

            // Storing ANFIS structure
            Rule [] ruleList = anfis.getRuleList();
            xmlWrite.writeStartElement("structure");
            xmlWrite.writeAttribute("inputs", "" + anfis.getInputCnt());
            xmlWrite.writeAttribute("activations", "" + anfis.getActivationCnt());
            xmlWrite.writeAttribute("rules", "" + ruleList.length);
            xmlWrite.writeEndElement();

            // Writing Activation Layer
            xmlWrite.writeStartElement("layer");
            xmlWrite.writeAttribute("id", "1");
            xmlWrite.writeAttribute("desc", "ACTIVATION");

            Activation[] activationList = anfis.getActivationList();
            for (int i = 0; i < anfis.getActivationCnt(); i++) {
                xmlWrite.writeStartElement("param");
                xmlWrite.writeAttribute("id", "" + (i + 1));
                xmlWrite.writeAttribute("MF", activationList[i].getMf().toString());

                // add parameters
                xmlWrite.writeStartElement("input");
                xmlWrite.writeAttribute("id", "" + (activationList[i].getInputNo() + 1));
                xmlWrite.writeEndElement();

                // add coefficients
                for (int j = 0; j < activationList[i].getParams().length; j++) {
                    xmlWrite.writeStartElement("coef");
                    xmlWrite.writeAttribute("id", "" + (j + 1));
                    xmlWrite.writeAttribute("val", "" + activationList[i].getParams()[j]);
                    xmlWrite.writeEndElement();
                }

                xmlWrite.writeEndElement();
            }
            xmlWrite.writeEndElement();

            // Writing Rule Layer
            xmlWrite.writeStartElement("layer");
            xmlWrite.writeAttribute("id", "2");
            xmlWrite.writeAttribute("desc", "RULE");

            for (int i = 0; i < ruleList.length; i++) {
                xmlWrite.writeStartElement("param");
                xmlWrite.writeAttribute("id", "" + (i + 1));
                xmlWrite.writeAttribute("ruleInputCnt", "" + ruleList[i].getInputActivations().length);
                xmlWrite.writeAttribute("OPERATION", ruleList[i].getOper().toString());

                // add inputs
                for (int j = 0; j < ruleList[i].getInputActivations().length; j++) {
                    xmlWrite.writeStartElement("input");
                    xmlWrite.writeAttribute("id", "" + (ruleList[i].getInputActivations()[j] + 1));
                    xmlWrite.writeEndElement();
                }

                xmlWrite.writeEndElement();
            }

            xmlWrite.writeEndElement();

            // Writing Linear (Consequent) Parameters
            xmlWrite.writeStartElement("consequent_coefs");
            xmlWrite.writeAttribute("count", "" + anfis.getLinearParamCnt());

            for (int i = 0; i < anfis.getLinearParamCnt(); i++) {
                xmlWrite.writeStartElement("consequent_coef");
                xmlWrite.writeAttribute("idx", "" + (i + 1));
                xmlWrite.writeAttribute("val", "" + anfis.getLinearParams()[i]);
                xmlWrite.writeEndElement();
            }

            xmlWrite.writeEndElement();

            // close document tag
            xmlWrite.writeEndDocument();
            xmlWrite.close();
        } catch (Exception ex) {
            System.out.println("Anfis.saveAnfisToFile(): " + filename);
        }

    }

    /**
     * Load Anfis structure and parameters from file (SAP parser is used).
     *
     * @param filename XML file with the ANFIS structure
     */
    public static Anfis loadAnfisFromFile(String filename) {
        Anfis anfis = null;
        try {
            System.out.println("Loading ANFIS from file: " + filename);
            File inputFile = new File(filename);
            SAXParserFactory factory = SAXParserFactory.newInstance();
            SAXParser saxParser = factory.newSAXParser();
            AnfisXmlHandler anfisXml = new AnfisXmlHandler();
            saxParser.parse(inputFile, anfisXml);

            anfis = new Anfis(anfisXml.inputCnt, anfisXml.activationCnt);
            anfis.setActivationList(anfisXml.getActivationList());
            anfis.setRuleList(anfisXml.getRuleList());
            anfis.setLinearParams(anfisXml.getLinearParams());
            anfis.init();
        } catch (Exception ex) {
            System.out.println("Anfis.loadAnfisFromFile(): " + ex);
            ex.printStackTrace();
        } finally {
            return anfis;
        }

    }


}

class AnfisXmlHandler extends DefaultHandler {
    ArrayList<Activation> activationList = new ArrayList<Activation>();
    ArrayList<Rule> ruleList = new ArrayList<Rule>();
    int[] ruleArr;

    Activation activation;
    double[] linearParams;

    int inputCnt = 0;
    int activationCnt = 0;
    int ruleCnt = 0;
    int ruleInputCnt = 0; // how many inputs rule has
    int layerId = 1;
    int activationId = 0;
    int ruleIdx = 0;
    String activationMF = "";
    String ruleOperation = "";

    @Override
    public void startElement(String uri,
                             String localName, String qName, Attributes attributes)
            throws SAXException {
        if (qName.equalsIgnoreCase("structure")) {
            inputCnt = Integer.parseInt(attributes.getValue("inputs"));
            activationCnt = Integer.parseInt(attributes.getValue("activations"));
            ruleCnt = Integer.parseInt(attributes.getValue("rules"));
            System.out.print("Input count: " + inputCnt + ", ");
            System.out.print("Activation count: " + activationCnt + ", ");
            System.out.println("Rule count: " + ruleCnt);
        } else if (qName.equalsIgnoreCase("layer")) {
            layerId = Integer.parseInt(attributes.getValue("id"));
            System.out.println("Layer ID : " + layerId);
        } else if (qName.equalsIgnoreCase("param")) {
            activationId = Integer.parseInt(attributes.getValue("id"));
            activationMF = attributes.getValue("MF");
            ruleOperation = attributes.getValue("OPERATION");
            if (layerId == 2) {
                ruleInputCnt = Integer.parseInt(attributes.getValue("ruleInputCnt"));
                ruleArr = new int[ruleInputCnt];
            }
            System.out.println("   Params : (" + activationId + "," + activationMF + "," + ruleOperation + ")");
        } else if (qName.equalsIgnoreCase("input")) {
            int inputId = Integer.parseInt(attributes.getValue("id"));
            System.out.println("      Input: " + inputId);
            // In case of Activation Layer
            if (layerId == 1) {
                activation = new Activation(inputId - 1, Activation.MembershipFunc.valueOf(activationMF));
            } else if (layerId == 2) {
                ruleArr[ruleIdx++] = inputId - 1; // because of Java array index
            }
        } else if (qName.equalsIgnoreCase("coef")) {
            int coefId = Integer.parseInt(attributes.getValue("id"));
            String val = attributes.getValue("val");
            if (val == null)
                activation.getParams()[coefId - 1] = Math.random();
            else
                activation.getParams()[coefId - 1] = Double.parseDouble(val);
            System.out.println("      Coef: (" + coefId + "," + activation.getParams()[coefId - 1] + ")");
        } else if (qName.equalsIgnoreCase("consequent_coefs")) {
            int paramCnt = Integer.parseInt(attributes.getValue("count"));
            linearParams = new double[paramCnt];
        } else if (qName.equalsIgnoreCase("consequent_coef")) {
            int paramIdx = Integer.parseInt(attributes.getValue("idx"));
            String val = attributes.getValue("val");
            if (val == null)
                linearParams[paramIdx - 1] = Math.random();
            else
                linearParams[paramIdx - 1] = Double.parseDouble(val);
            System.out.println("      Consequent Coef: (" + paramIdx + "," + linearParams[paramIdx - 1] + ")");
        }
    }

    @Override
    public void endElement(String uri,
                           String localName, String qName) throws SAXException {
        if (qName.equalsIgnoreCase("layer")) {
            System.out.println("/Layer");
        } else if (qName.equalsIgnoreCase("param")) {
            if (layerId == 1) {
                // add Activation object to a list
                activationList.add(activation);
            }
            if (layerId == 2) {
                // add Rule object to a list
                Rule rule = new Rule(ruleArr, Rule.RuleOperation.valueOf(ruleOperation));
                ruleList.add(rule);
                ruleIdx = 0;
                ruleArr = new int[ruleCnt];
            }
        }
    }

    public int getInputCnt() {
        return inputCnt;
    }

    public int getActivationCnt() {
        return activationCnt;
    }

    public double[] getLinearParams() {
        return linearParams;
    }

    public Activation[] getActivationList() {
        return activationList.toArray(new Activation[activationList.size()]);
    }

    public Rule[] getRuleList() {
        return ruleList.toArray(new Rule[ruleList.size()]);
    }

}