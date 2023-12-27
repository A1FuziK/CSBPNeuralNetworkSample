// See https://aka.ms/new-console-template for more information

using CSBPNeuralNetworkSample;

static void TeachAndTest()
{
    // Initialize variables
    BPNeuralNetwork bpNetwork = new BPNeuralNetwork(0.1, 0.5, 4, new int[] { 4, 64, 64, 16 });

    //Input data width "should" fit with the input bits.
    //The lines number should fit with the lines of the out data (requested result).
    double[][] inputData = new double[][] {
        new double[]{0,0,0,0},
        new double[]{0,0,0,1},
        new double[]{0,0,1,0},
        new double[]{0,0,1,1},
        new double[]{0,1,0,0},
        new double[]{0,1,0,1},
        new double[]{0,1,1,0},
        new double[]{0,1,1,1},
        new double[]{1,0,0,0},
        new double[]{1,0,0,1},
        new double[]{1,0,1,0},
        new double[]{1,0,1,1},
        new double[]{1,1,0,0},
        new double[]{1,1,0,1},
        new double[]{1,1,1,0},
        new double[]{1,1,1,1}
    };

    //The out data contains the requested result from the network.
    //So this is the target. First line of the input should ends up
    //as the first line of the out data.
    //The weight will be (hopefully) adjusted to make it happen.
    double[][] outData = new double[][] {
        new double[]{1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        new double[]{0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        new double[]{0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
        new double[]{0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},
        new double[]{0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
        new double[]{0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
        new double[]{0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},
        new double[]{0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0},
        new double[]{0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0},
        new double[]{0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
        new double[]{0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},
        new double[]{0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0},
        new double[]{0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0},
        new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0},
        new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0},
        new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}
    };
    
    int maxIterations = 2000000;
    int iterations;
    bool bTrained = false;
    int trainCycles = 0;
    double comulatedDistance = 0.0;
    double errorThreashold = 0.00001;
    double dTemp = 0.0;

    Console.WriteLine("Training the NeuralNetwork...");

    for (iterations = 0; iterations < maxIterations && !bTrained; iterations++)
    {
        bTrained = true;

        for (int iBuffLearn = 0; iBuffLearn < 16; iBuffLearn++)
        {
            dTemp = 0.0;

            for (int iInnerLeanCycle = 0; iInnerLeanCycle < 10; iInnerLeanCycle++)
            {
                //Do a back propagation process (learning) for this sample.
                //First it push the input through the network.
                //Then it calculates the difference between the results
                //and the expected results. It makes tiny changes on the
                //weights every time.
                bpNetwork.BackPropagate(inputData[iBuffLearn], outData[iBuffLearn]);
                
                //Get the error rate for this sample.
                dTemp = bpNetwork.MeanSquareError(outData[iBuffLearn]);

                comulatedDistance += dTemp;
                trainCycles++;
                if (dTemp <= (comulatedDistance / trainCycles) || dTemp < errorThreashold)
                    break;
            }

            if (dTemp >= errorThreashold)
            {
                //If any training sample fails - let's continue the learning.
                bTrained = false;
            }
        }
        
        if (iterations % (maxIterations / 100) == 0)
        {
            Console.WriteLine("Still training... last meanSquareError: " +
                dTemp + " average meanSquareError: " +
                (comulatedDistance / trainCycles));
        }
    }

    Console.WriteLine($"{trainCycles} trainCycle completed... in " +
        $"{iterations} iteration cycles... " +
        " average meanSquareError: " + (comulatedDistance / trainCycles));

    string weightString = bpNetwork.GetNetWeights();
    //Console.WriteLine("begin weightString");
    //Console.WriteLine(weightString);
    //Console.WriteLine("end weightString");

    bpNetwork = new BPNeuralNetwork(0.1, 0.5, 4, new int[] { 4, 64, 64, 16 });
    bpNetwork.SetNetWeights(weightString);

    Console.WriteLine("Testing the network...");

    for (int iBuffLearn = 0; iBuffLearn < 16; iBuffLearn++)
    {
        bpNetwork.FeedForward(inputData[iBuffLearn]);

        string resultMessage = iBuffLearn.ToString() + "\t";

        for (int iOut = 0; iOut < 16; iOut++)
        {
            double dOut = bpNetwork.OutValue(iOut);

            if (dOut >= 1.1)
                resultMessage += "H ";
            else if (dOut >= 0.8)
                resultMessage += "1 ";
            else if (dOut <= 0.2)
                resultMessage += "0 ";
            else if (dOut <= 0)
                resultMessage += "L ";
            else 
                resultMessage += "M ";
        }

        Console.WriteLine(resultMessage);
    }
}

static void TestOnly()
{
    var bpNetwork = new BPNeuralNetwork(0.1, 0.5, 4, new[] { 4, 8, 8, 16 });

    string weightString = "3.133351190970881;-0.6826674273344457;-1.929090000860336;-4.1004252334900215;2.4538816894253612;1.1535679543007813;-2.6751575667099203;3.538179051276173;0.4134143459022974;-0.08283963886440938;-0.060027511307136905;-0.5319063364180328;0.2962554663706748;-0.18298960084805893;-0.4349381567021938;-0.6568479363036426;-0.21160075190004957;0.1138222561268581;0.18176169914549808;-0.2977328723684596;0.02978697552668974;-0.8047480304124859;-0.5050464529016718;-0.6295082487306058;-0.2453497790900534;-2.475440061675393;2.596770110273842;-1.8090164628925305;-2.2155736620202373;4.109144680024236;-0.9222514924337086;-0.07488308918505515;0.32964154114638156;-0.37372051510451376;-0.901080617036282;-2.7630718651987025;0.9448787690918576;1.3419249243578746;3.030609865045329;-1.339096274410195;-0.32211745837563305;0.47543955721595565;0.23469082234727767;-0.017334253333639277;0.9688453962751296;1.4356150902675948;-0.8796989148445671;-2.280548981581638;4.938699896524856;3.908124696601413;2.307887793910971;0.7098714708390332;0.7818765245886184;0.47268880812118086;-0.3848357292538665;0.10124655114327163;2.250739530109724;-0.32326266147356725;1.7182857923775092;0.3995127518783098;-0.4466644715772592;0.9772704395825436;0.024649673037484998;-1.6847539618100271;-0.8770577609800081;-0.7233498231589159;-1.8162417290948136;1.1315044007083288;-0.28691595646320556;-0.7779526442181597;-0.9798474523085927;-0.5277959978067184;3.524124086769119;-0.7267199425407544;-1.4826540817337148;-1.1548413680744;-0.2108123047653791;0.008507557897185285;0.7066417076877771;-0.7739438117009789;0.9881408744525086;-0.786569124299429;0.32071788251027344;-0.8266827498755084;-0.06963284558653308;-1.2075044972917304;-2.3859619751298835;0.39020907676508143;0.7761678484480571;-0.7417999606377745;3.2274175795272546;-0.7087989160961214;1.5123488147882747;0.407673488568874;-3.3030716765100565;2.85224810397152;0.9792100856637593;-0.5905802618883244;-0.16571864053594032;1.3092010549793982;-0.002093852838165766;-0.08660945156909208;2.7237369159214273;-2.2430685139502216;1.8930167847137294;0.80906710071313;0.6841439145043373;0.036500860651621725;2.9130170838456877;-0.06627104175102705;-2.4937706698113464;-2.0313497124641287;0.203133501709919;-1.4850544054922046;-0.04555221740664171;0.6211764459422178;-0.8166989884592297;0.7064720237211323;-4.054850397908444;-0.7294130233169749;-0.18924720951628954;0.8722913912460015;-2.269954823235917;0.17978685500435318;-1.8120134815569804;0.26280899222903054;1.1351939309628654;0.7009502140703096;-2.4267017126233084;0.6023525033671092;0.8684850689202243;-2.2469008883163855;0.6868176539274131;-0.2424610741359286;-0.8889980092493922;-1.9404627373210108;0.16955346505643823;1.6352145185713254;0.2298045264938271;-2.9038314101988685;0.10532107888056204;0.7507854756759537;-0.975422155066597;-0.3408196246761691;-1.2541102927858252;0.35872578628615664;-0.7231091615381442;-0.043073876247130445;-0.5833429538311496;-2.5196233906049383;-0.16960256204565755;0.6663172210517151;-0.33837775417577654;0.4322862964416411;-2.153702749869823;1.0282891452140661;-1.6336237179280388;-2.174609636715976;-2.02334564180038;0.834331796692054;0.659823346940563;-0.023613760783992386;1.6215790340018288;-0.9196802770278225;-0.8929887185226236;-0.28763558351996305;-0.07873047089339869;-3.246020761779475;0.8528140568297227;-0.4627244947965969;-0.5146729445035509;-0.07357563281937787;0.8074926165286977;1.0247496676557977;-0.007304708404348731;-2.5891741406801465;-0.1787242117098646;-0.026147596625591827;-1.298168574432098;-0.005130818093731168;1.2339753105773625;-0.34166968398188063;-0.006740404071327206;-0.21346059488189287;-1.197198809902388;0.10879641133436636;0.9530088623471833;0.7240808478727502;0.663375813037879;-2.023662462513818;-2.2825149295815548;-0.40711584744803603;-1.6538616350935154;1.89933977466493;-0.1853361352148542;-0.776759309953611;-2.0405048286671033;0.46713882597404954;-0.9376167164400584;-1.3903872002450683;0.49633023810953686;0.6318985072419454;-1.8397592307304478;0.6134432628747148;1.7315065600827102;-1.9088285208807723;-0.8379478106052397;-0.7746629394364418;-1.315120195159458;-0.17920287713110078;-0.8572449569147962;-1.0570763593272974;-1.289441311204801;-0.4079431265797597;-2.5856224316745404;0.30087251310258634;-1.0092933504440784;0.9165623173682547;1.9755076836908503;-0.9129459631905664;-1.126750465210009;-0.3020484794734461;-1.4526127892071536;1.4636635312633903;0.17477550245645448;-0.5894950290391542;-2.284888239758663;-2.085151733133984;-1.2687069827226556;2.219243920639292;-2.2097861403897134;0.7521441598643576;-0.025244494155107666;0.27988909575719795;-0.5067590367796894;-0.4185564918530939;-1.2780218589729422;-0.3632205142400126;0.8179504270905709;-0.5639730800337975;-1.6577627645944675;1.0128593506179455;0.11828747876001686;-2.821639136798806;-0.854857793462443;-0.45885059353015056;-1.1546277774089493;1.0550789675316705;-1.6438391468681939;-0.21397686087940643;-1.557982482980854;-0.5878390878088701;-1.3061343596814905;1.3964589059391253;-2.58088490995498;0.1627850502703333;";

    bpNetwork.SetNetWeights(weightString);

    double[][] inputData = new double[][]
    {
        new double[] {0,0,0,0},
        new double[] {0,0,0,1},
        new double[] {0,0,1,0},
        new double[] {0,0,1,1},
        new double[] {0,1,0,0},
        new double[] {0,1,0,1},
        new double[] {0,1,1,0},
        new double[] {0,1,1,1},
        new double[] {1,0,0,0},
        new double[] {1,0,0,1},
        new double[] {1,0,1,0},
        new double[] {1,0,1,1},
        new double[] {1,1,0,0},
        new double[] {1,1,0,1},
        new double[] {1,1,1,0},
        new double[] {1,1,1,1}
    };

    Console.WriteLine("Testing the network...");

    for(int iBuffLearn = 0; iBuffLearn < 16; iBuffLearn++)
    {
        bpNetwork.FeedForward(inputData[iBuffLearn]);

        string resultMessage = iBuffLearn.ToString() + "\t";

        for(int iOut = 0; iOut < 16; iOut++)
        {
            double dOut = bpNetwork.OutValue(iOut);

            if(dOut >= 1.1) {
                resultMessage += "H ";
            }
            else if(dOut >= 0.8) {
                resultMessage += "1 ";
            }
            else if(dOut <= 0.2) {
                resultMessage += "0 ";
            }
            else if(dOut <= 0) {
                resultMessage += "L ";
            }
            else {
                resultMessage += "M ";
            }
        }

        Console.WriteLine(resultMessage);
    }
}

TeachAndTest();

TestOnly();
