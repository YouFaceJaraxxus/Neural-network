using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_network
{
    class Program
    {
        public static int MID_LAYER = 20;
        public static int bigCount = 0;
        public static double lossMargin = 0.01;

        public static double HOOP_RADIUS = 0.2286;
        public static double THROWER_HEIGHT = 2.62;
        public static double BALL_RADIUS = 0.12;
        public static double HOOP_HEIGHT = 3.05;
        public static double HOOP_BAR_RADIUS = 0.02;
        public static double VERTICAL_PADDING = 0.05;
        public static double VERTICAL_BALL_MAX = HOOP_HEIGHT + VERTICAL_PADDING;
        public static double VERTICAL_BALL_MIN = HOOP_HEIGHT - VERTICAL_PADDING;
        public static double HORIZONTAL_BALL_DIFF = 0.108;
        public static  double PADDING = 0.005;

        public static double[] maxsX = new double[3]; //3 inputs
        public static double[] minsX = new double[3];

        public static double[] maxsY = new double[2];  //2 outputs
        public static double[] minsY = new double[2];

        static Program()
        {
            maxsX[0] = 18;
            minsX[0] = 6.7;
            maxsX[1] = 3;
            minsX[1] = 1;
            maxsX[2] = 2.9;
            minsX[2] = 2;
            maxsY[0] = 40;
            minsY[0] = 35;
            maxsY[1] = 14;
            minsY[1] = 8;
        }
        /*hoop diameter is 45.72 , radius is (1/2 = 22.86~22.8), 12 is the ball radius, so the diff = 22.8-12 = 10.8cm
        this is the amount of space that the center of the ball can be positioned away from the hoop bars
        and not touch any of them*/

        public static void Main(string[] args)
        {
            double[][] dataTrainX = readMatrixFromCSV("smaller_dataset\\dataX_v2.csv");
            double[][] dataTrainY = readMatrixFromCSV("smaller_dataset\\dataY_v2.csv");
            double[][] dataTestX;
            double[][] dataTestY;
            double learningFactor = 5;
            double errorMargin = 0.00001;

            bool train = true;
            if(train)
            {
                for (int i = 0; i < dataTrainX.Length; i++)
                {
                    double distanceToHoop = dataTrainX[i][0];
                    double distanceToBlocker = dataTrainX[i][1];
                    double blockerHeight = dataTrainX[i][2];

                    double angle = dataTrainY[i][0];
                    double speed = dataTrainY[i][1];

                    if (!goodThrow(distanceToHoop, distanceToBlocker, blockerHeight, DegreeToRadian(angle), speed, true))
                    {
                        Console.WriteLine("ERROR!");
                    }
                }

                double[][][][] split = splitData(5000, dataTrainX, dataTrainY);
                dataTrainX = split[0][0];
                dataTrainY = split[0][1];
                dataTestX = split[1][0];
                dataTestY = split[1][1];

                int testCount = 5000;
                double[][] dataTestX_tmp = new double[testCount][];
                double[][] dataTestY_tmp = new double[testCount][];

                for (int i = 0; i < testCount; i++)
                {
                    dataTestX_tmp[i] = dataTestX[i];
                    dataTestY_tmp[i] = dataTestY[i];
                }

                dataTestX = dataTestX_tmp;
                dataTestY = dataTestY_tmp;


                scaleMatrix(dataTrainX, minsX, maxsX);
                scaleMatrix(dataTrainY, minsY, maxsY);
                scaleMatrix(dataTestX, minsX, maxsX);
                scaleMatrix(dataTestY, minsY, maxsY);

                shuffleArray(dataTrainX, dataTrainY, dataTrainX.Length);
                shuffleArray(dataTestX, dataTestY, dataTestX.Length);



                double[][] W1 = new double[MID_LAYER][];
                double[][] W2 = new double[MID_LAYER][];
                double[][] W3 = new double[2][];

                initWeights(W1, W2, W3);


                double[][][] result = trainNetwork(dataTrainX, dataTrainY, W1, W2, W3, errorMargin, learningFactor, 2000, dataTestX, dataTestY);


                saveMatrixAsCSV("smaller_dataset\\w1.csv", result[0]);
                saveMatrixAsCSV("smaller_dataset\\w2.csv", result[1]);
                saveMatrixAsCSV("smaller_dataset\\w3.csv", result[2]);
                Console.ReadLine();
            }
            else
            {
                double[][] W1 = readMatrixFromCSV("smaller_dataset\\w1_best.csv");
                double[][] W2 = readMatrixFromCSV("smaller_dataset\\w2_best.csv");
                double[][] W3 = readMatrixFromCSV("smaller_dataset\\w3_best.csv");
                scaleMatrix(dataTrainX, minsX, maxsX);
                scaleMatrix(dataTrainY, minsY, maxsY);
                shuffleArray(dataTrainX, dataTrainY, dataTrainX.Length);

                int good = 0;
                PADDING = 0.05;

                for (int i = 0; i < dataTrainX.Length; i+=3)
                {
                    double[] middleNet_Q2 = new double[MID_LAYER];
                    double[] middleY_Q2 = new double[MID_LAYER];
                    double[] middleNet_Q3 = new double[MID_LAYER];
                    double[] middleY_Q3 = new double[MID_LAYER];
                    double[] outputNet = new double[2];
                    double[] outputY = new double[2];

                    double[] input_test = dataTrainX[i];

                    calculateForward(W1, input_test, middleNet_Q2);  //W1 X input
                    doSigmoidToArray(middleNet_Q2, middleY_Q2); //sigmoids to net_q2
                    calculateForward(W2, middleY_Q2, middleNet_Q3); //W2 X Y_q2
                    doSigmoidToArray(middleNet_Q3, middleY_Q3); //sigmoids to net_q3
                    calculateForward(W3, middleY_Q3, outputNet);    //W3 X Y_q3
                    doSigmoidToArray(outputNet, outputY);   //sigmoids to net_output

                    double angle = outputY[0];

                    double speed = outputY[1];

                    double scaledAngle = scaleOut(angle, minsY[0], maxsY[0]);
                    double scaledSpeed = scaleOut(speed, minsY[1], maxsY[1]);
                    double scaledDistanceToHoop = scaleOut(input_test[0], minsX[0], maxsX[0]);
                    double scaledDistanceToBlocker = scaleOut(input_test[1], minsX[1], maxsX[1]);
                    double scaledBlockerHeight = scaleOut(input_test[2], minsX[2], maxsX[2]);

                    if (goodThrow(scaledDistanceToHoop, scaledDistanceToBlocker, scaledBlockerHeight, DegreeToRadian(scaledAngle), scaledSpeed, false))
                    {
                        /*Console.WriteLine("ANGLE: " + scaledAngle);
                        Console.WriteLine("SPEED: " + scaledSpeed);*/
                        good++;
                    }

                }

                Console.WriteLine("GOOD: " + good);
                Console.WriteLine("TOTAL: " + dataTrainX.Length/3);
                Console.WriteLine("RATIO: " + ((double)good*3 / dataTrainX.Length));
            }

            Console.ReadLine();

        }

        private static bool goodError(double errorMargin, double[] wantedOutput, double[] actualOutput)
        {
            double tempError = 0.0;
            for (int i = 0; i < wantedOutput.Length; i++)
            {
                tempError += Math.Pow((wantedOutput[i] - actualOutput[i]), 2);
            }
            tempError /= 2;
            return tempError < errorMargin;
        }

        private static bool goodLoss(double lossMargin, double[] wantedOutput, double[] actualOutput)
        {
            double tempError = 0.0;
            for (int i = 0; i < wantedOutput.Length; i++)
            {
                tempError += Math.Pow((wantedOutput[i] - actualOutput[i]), 2);
            }
            tempError /= 2;
            if (bigCount > 10)
            {
                //bigCount = bigCount;
            }
            return tempError < lossMargin;
        }
        private static double[][] readMatrixFromCSV(string path)
        {
            string[] fileLines = File.ReadAllLines(path);
            double[][] numbers = new double[fileLines.Length][];
            for (int i = 0; i < fileLines.Length; i++)
            {
                string[] numbersString = fileLines[i].Split(',');
                double[] row = new double[numbersString.Length];
                for (int j = 0; j < numbersString.Length; j++)
                {
                    row[j] = Double.Parse(numbersString[j]);
                }
                numbers[i] = row;
            }
            return numbers;
        }

        private static double[] readArrayFromCSV(string path)
        {
            string[] fileLines = File.ReadAllLines(path);
            double[] numbers = new double[fileLines.Length];
            for (int i = 0; i < fileLines.Length; i++)
            {
                numbers[i] = Double.Parse(fileLines[i]);
            }
            return numbers;
        }

        private static void calculateForward(double[][] weights, double[] inputs, double[] neurons)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                double res = 0;
                for (int j = 0; j < weights[i].Length; j++) res += weights[i][j] * inputs[j];
                neurons[i] = res;
            }
        }

        private static void calculateForwardWithBias(double[][] weights, double[] biases, double[] inputs, double[] neurons)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                double res = 0;
                for (int j = 0; j < weights[i].Length; j++) res += weights[i][j] * inputs[j];
                res += biases[i];
                neurons[i] = res;
            }
        }

        private static void doSigmoidToArray(double[] inputs, double[] outputs)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                outputs[i] = Sigmoid(inputs[i]);
            }
        }
        private static void calculateForwardWithSigma(double[][] weights, double[] biases, double[] inputs, double[] neurons)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                double res = 0;
                for (int j = 0; j < weights[i].Length; j++) res += weights[i][j] * inputs[j];
                res += biases[i];
                neurons[i] = Sigmoid(res);
            }
        }
        private static double scaleIn(double val, double min, double max)
        {
            return (val - min) / (max - min);
        }

        private static double scaleOut(double val, double min, double max)
        {
            return (max - min) * val + min;
        }


        private static void saveArrayAsCSV(string path, double[] array)
        {
            string[] lines = new string[array.Length];
            for (int i = 0; i < array.Length; i++)
            {
                lines[i] = array[i].ToString();
            }
            File.WriteAllLines(path, lines);
        }
        private static void saveMatrixAsCSV(string path, double[][] matrix)
        {
            string[] lines = new string[matrix.Length];
            for (int i = 0; i < matrix.Length; i++)
            {
                double[] row = matrix[i];
                string line = "";
                for (int j = 0; j < row.Length - 1; j++)
                {
                    line = line + (row[j].ToString() + ",");
                }
                line = line + row[row.Length - 1].ToString();
                lines[i] = line;
            }
            File.WriteAllLines(path, lines);
        }

        private static float Sigmoid(double value)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-value));
        }

        private static double SigmoidPrime(double value)
        {
            double sigmoid = Sigmoid(value);
            return sigmoid * (1 - sigmoid);
        }

        private static double[][][][] splitData(int numTraining, double[][] dataX, double[][] dataY) //the return[0] is training data, the return[1] is test data, return[x][0] is input, return [x][1] is output
        {
            double[][][][] splitData = new double[2][][][]; //two kinds, train and test data
            splitData[0] = new double[2][][]; //two kinds, input and output data
            splitData[1] = new double[2][][]; //two kinds, input and output data
            splitData[0][0] = new double[numTraining][]; //training input data
            splitData[0][1] = new double[numTraining][]; //training output data
            splitData[1][0] = new double[dataX.Length - numTraining][];    //test input data
            splitData[1][1] = new double[dataX.Length - numTraining][];    //test output data
            List<int> testers = new List<int>(dataX.Length - numTraining); //dataX length == dataY length
            Random rand = new Random();
            for (int i = 0; i < dataX.Length - numTraining; i++)
            {
                bool found = false;
                while (!found)
                {
                    int nextInt = rand.Next(dataX.Length);
                    if (!testers.Contains(nextInt))
                    {
                        found = true;
                        testers.Add(nextInt);
                    }
                }
            }
            int trainCount = 0;
            int testCount = 0;
            for (int i = 0; i < dataX.Length; i++)
            {
                if (!testers.Contains(i))
                {
                    splitData[0][0][trainCount] = dataX[i];
                    splitData[0][1][trainCount++] = dataY[i];
                }
                else
                {
                    splitData[1][0][testCount] = dataX[i];
                    splitData[1][1][testCount++] = dataY[i];
                }
            }
            return splitData;
        }

        private static double getSingleCost(double[] actualVals, double[] expectedVals)
        {
            double cost = 0.0;
            for (int i = 0; i < actualVals.Length; i++)
            {
                cost += Math.Pow(actualVals[i] - expectedVals[i], 2);
            }
            return cost;
        }

        private static void scaleMatrix(double[][] array, double[] mins, double[] maxs)
        {
            for (int i = 0; i < array.Length; i++)
            {
                for (int j = 0; j < mins.Length; j++)
                {
                    array[i][j] = scaleIn(array[i][j], mins[j], maxs[j]);
                }
            }
        }

        private static void shuffleArray(double[][] arrayX, double[][] arrayY, int size)
        {
            Random rand = new Random();
            List<int> unusedIndexes = new List<int>();
            for (int i = 0; i < size; i++) unusedIndexes.Add(i);
            int currentIndex = 0;
            do
            {
                int index = rand.Next(unusedIndexes.Count());
                unusedIndexes.RemoveAt(index);
                double[] tmp = arrayX[index];
                arrayX[index] = arrayX[currentIndex];
                arrayX[currentIndex] = tmp;
                tmp = arrayY[index];
                arrayY[index] = arrayY[currentIndex];
                arrayY[currentIndex] = tmp;
                currentIndex++;
            }
            while (unusedIndexes.Count() > 0);
        }

        private static void copyMatrix(double[][] source, double[][] destination)
        {
            for(int i=0;i<source.Length;i++)
            {
                for(int j=0;j<source[i].Length;j++)
                {
                    destination[i][j] = source[i][j];
                }
            }
        }

        private static double[][][] trainNetwork(double[][] dataTrainX, double[][] dataTrainY, double[][] W1, double[][] W2, double[][] W3, double errorMargin, double learningFactor, int numIterations, double[][] dataTestX, double[][] dataTestY)
        {
            double[][][] result = new double[3][][]; //W1, W2, W3
            double[] deltaQ_4 = new double[2];  //deltas for W3
            double[] deltaQ_3 = new double[MID_LAYER]; //deltas for W2
            double[] deltaQ_2 = new double[MID_LAYER]; //deltas for W1

            double[][] W1_best = new double[MID_LAYER][];
            double[][] W2_best = new double[MID_LAYER][];
            double[][] W3_best = new double[2][];
            initWeights(W1_best, W2_best, W3_best);
            int count_best = 0;

            double[] middleNet_Q2 = new double[MID_LAYER];
            double[] middleY_Q2 = new double[MID_LAYER];
            double[] middleNet_Q3 = new double[MID_LAYER];
            double[] middleY_Q3 = new double[MID_LAYER];
            double[] outputNet = new double[2];
            double[] outputY = new double[2];
            for (int i = 0; i < numIterations; i++)
            {
                if (i % 30 == 0) learningFactor /= 1.5;
                if (i % 10 == 0)
                {
                    Console.WriteLine(i);
                    int good = 0;

                    for (int j = 0; j < dataTestX.Length; j++)
                    {
                        double[] input_test = dataTestX[j];

                        calculateForward(W1, input_test, middleNet_Q2);  //W1 X input
                        doSigmoidToArray(middleNet_Q2, middleY_Q2); //sigmoids to net_q2
                        calculateForward(W2, middleY_Q2, middleNet_Q3); //W2 X Y_q2
                        doSigmoidToArray(middleNet_Q3, middleY_Q3); //sigmoids to net_q3
                        calculateForward(W3, middleY_Q3, outputNet);    //W3 X Y_q3
                        doSigmoidToArray(outputNet, outputY);   //sigmoids to net_output

                        double angle = outputY[0];

                        double speed = outputY[1];

                        double scaledAngle = scaleOut(angle, minsY[0], maxsY[0]);
                        double scaledSpeed = scaleOut(speed, minsY[1], maxsY[1]);
                        double scaledDistanceToHoop = scaleOut(input_test[0], minsX[0], maxsX[0]);
                        double scaledDistanceToBlocker = scaleOut(input_test[1], minsX[1], maxsX[1]);
                        double scaledBlockerHeight = scaleOut(input_test[2], minsX[2], maxsX[2]);

                        if (goodThrow(scaledDistanceToHoop, scaledDistanceToBlocker, scaledBlockerHeight, DegreeToRadian(scaledAngle), scaledSpeed, false))
                        {
                            /*Console.WriteLine("ANGLE: " + scaledAngle);
                            Console.WriteLine("SPEED: " + scaledSpeed);*/
                            good++;
                        }

                    }

                    Console.WriteLine("GOOD: " + good);
                    Console.WriteLine("TOTAL: " + dataTestX.Length);
                    Console.WriteLine("RATIO: " + ((double)good / dataTestX.Length));
                    if(good>count_best)
                    {
                        count_best = good;
                        copyMatrix(W1, W1_best);
                        copyMatrix(W2, W2_best);
                        copyMatrix(W3, W3_best);
                        saveMatrixAsCSV("smaller_dataset\\w1_best.csv", W1);
                        saveMatrixAsCSV("smaller_dataset\\w2_best.csv", W2);
                        saveMatrixAsCSV("smaller_dataset\\w3_best.csv", W3);
                    }
                }
                for (int j = 0; j < dataTrainX.Length; j++)
                {
                    double[] input = dataTrainX[j];
                    double[] wantedOutput = dataTrainY[j];

                    
                    calculateForward(W1, input, middleNet_Q2);  //W1 X input
                    doSigmoidToArray(middleNet_Q2, middleY_Q2); //sigmoids to net_q2
                    calculateForward(W2, middleY_Q2, middleNet_Q3); //W2 X Y_q2
                    doSigmoidToArray(middleNet_Q3, middleY_Q3); //sigmoids to net_q3
                    calculateForward(W3, middleY_Q3, outputNet);    //W3 X Y_q3
                    doSigmoidToArray(outputNet, outputY);   //sigmoids to net_output

                    double angle = outputY[0];
                    double speed = outputY[1];

                    double scaledAngle = scaleOut(angle, minsY[0], maxsY[0]);
                    double scaledSpeed = scaleOut(speed, minsY[1], maxsY[1]);
                    double scaledDistanceToHoop = scaleOut(input[0], minsX[0], maxsX[0]);
                    double scaledDistanceToBlocker = scaleOut(input[1], minsX[1], maxsX[1]);
                    double scaledBlockerHeight = scaleOut(input[2], minsX[2], maxsX[2]);

                    if (!goodThrow(scaledDistanceToHoop, scaledDistanceToBlocker, scaledBlockerHeight, DegreeToRadian(scaledAngle), scaledSpeed, true))
                    {
                        //fix W3
                        for (int k = 0; k < outputY.Length; k++)   //iterate 2 output layer neurons
                        {
                            deltaQ_4[k] = (wantedOutput[k] - outputY[k]) * SigmoidPrime(outputNet[k]);
                        }
                        for (int k = 0; k < outputY.Length; k++) //iterate W2 and apply appropriate change (2x10 W2 matrix)
                        {
                            for (int l = 0; l < middleY_Q3.Length; l++)
                            {
                                W3[k][l] += learningFactor * deltaQ_4[k] * middleY_Q3[l];
                            }
                        }

                        //fix W2
                        for (int k = 0; k < middleY_Q3.Length; k++)   //iterate 10 middle layer neurons Q3
                        {
                            double sum = 0.0;
                            for (int l = 0; l < deltaQ_4.Length; l++) sum += W3[l][k] * deltaQ_4[l]; //iterate 2 Q4 deltas
                            deltaQ_3[k] = SigmoidPrime(middleNet_Q3[k]) * sum;
                        }
                        for (int k = 0; k < middleY_Q3.Length; k++) //iterate W2 and apply appropriate change (10x10 W2 matrix)
                        {
                            for (int l = 0; l < middleY_Q2.Length; l++)
                            {
                                W2[k][l] += learningFactor * deltaQ_3[k] * middleY_Q2[l];
                            }
                        }

                        //fix W1
                        for (int k = 0; k < middleY_Q2.Length; k++)   //iterate 10 middle layer neurons Q2
                        {
                            double sum = 0.0;
                            for (int l = 0; l < deltaQ_3.Length; l++) sum += W2[l][k] * deltaQ_3[l]; //iterate 10 Q3 deltas
                            deltaQ_2[k] = SigmoidPrime(middleNet_Q2[k]) * sum;
                        }
                        for (int k = 0; k < middleY_Q2.Length; k++) //iterate W2 and apply appropriate change (10x10 W2 matrix)
                        {
                            for (int l = 0; l < input.Length; l++)
                            {
                                W1[k][l] += learningFactor * deltaQ_2[k] * input[l];
                            }
                        }
                    }

                }
            }
            result[0] = W1_best;
            result[1] = W2_best;
            result[2] = W3_best;
            return result;
        }

        private static double Logit(double value)
        {
            return Math.Log((value) / (1 - value));
        }
        private static void initWeights(double[][] W1, double[][] W2, double[][] W3)
        {
            //our first weight array is 3x10, and first bias is 3x1
            for (int i = 0; i < MID_LAYER; i++)
            {
                Random rand = new Random();
                W1[i] = new double[3];
                for (int j = 0; j < 3; j++)
                {
                    W1[i][j] = (rand.NextDouble() * 6) - 3; //weights are from -3 to 3
                }
            }

            //our second weight array is 10x10, and second bias is 10x1
            for (int i = 0; i < MID_LAYER; i++)
            {
                Random rand = new Random();
                W2[i] = new double[MID_LAYER];
                for (int j = 0; j < 10; j++)
                {
                    W2[i][j] = (rand.NextDouble() * 6) - 3; //weights are from -3 to 3
                }
            }

            //our second weight array is 2x10, and second bias is 10x1
            for (int i = 0; i < 2; i++)
            {
                Random rand = new Random();
                W3[i] = new double[MID_LAYER];
                for (int j = 0; j < 10; j++)
                {
                    W3[i][j] = (rand.NextDouble() * 6) - 3; //weights are from -3 to 3
                }
            }
        }


        private static double DegreeToRadian(double angle)
        {
            return Math.PI * angle / 180.0;
        }


        private static double RadianToDegree(double angle)
        {
            return angle * (180.0 / Math.PI);
        }


        private static double g = 9.81;
        private static double parabola(double x, double v, double angle, double throwerHeight)  //angle is in radians
        {
            /*Console.WriteLine(angle);
            Console.WriteLine(x);
            Console.WriteLine(Math.Tan(angle));
            Console.WriteLine(Math.Cos(angle));*/

            return x * Math.Tan(angle) - (g / 2) * ((Math.Pow(x, 2)) / (Math.Pow(v, 2) * Math.Pow(Math.Cos(angle), 2))) + throwerHeight;
        }


        public static double circleRad(double r, double x, double xo, double yo)
        {
            return Math.Sqrt(Math.Pow(r, 2) - Math.Pow(x - xo, 2)) + yo;
        }

        public static double dotDistance(double x1, double y1, double x2, double y2)
        {
            double xVal = Math.Pow(x1 - x2, 2);
            double yVal = Math.Pow(y1 - y2, 2);
            double retVal = Math.Sqrt(xVal + yVal);
            return retVal;
        }

        public static bool circleCrossDot(double x1, double y1, double r, double x2, double y2) //determines whether a circle(x1, y1, r) touches or crosses a dot (x2, y2)
        {
            double minDist = dotDistance(x1, y1, x2, y2);
            return minDist < r;
        }

        public static bool circlesCross(double x1, double y1, double r1, double x2, double y2, double r2) //ddetermines whether 2 circles(x1, y1, r1), (x2, y2, r2) touch or intersect
        {
            double minDist = dotDistance(x1, y1, x2, y2);
            return minDist < (r1 + r2);
        }

        public static bool goodPos(double x, double y, double goodPosXMin, double goodPosXMax, double goodPosYMin, double goodPosYMax, bool precise) //determines if the ball has reached the final wanted position
        {
            if (precise) return x < goodPosXMax && x > goodPosXMin && y < goodPosYMax && y > goodPosYMin;
            else
            {
                return x < goodPosXMax + PADDING && x > goodPosXMin - PADDING && y < goodPosYMax + PADDING && y > goodPosYMin - PADDING;
            }
        }

        public static bool goodThrow(double distanceToHoop, double distanceToBlocker, double blockerHeight, double angle, double speed, bool precise)
        {
            double goodPosXMax = distanceToHoop + HORIZONTAL_BALL_DIFF;
            double goodPosXMin = distanceToHoop - HORIZONTAL_BALL_DIFF;
            double hoopX1 = distanceToHoop - HOOP_RADIUS;
            double hoopX2 = distanceToHoop + HOOP_RADIUS;

            for (double x = distanceToBlocker - BALL_RADIUS; x < distanceToBlocker + BALL_RADIUS; x = x + 0.01)
            {
                double y = parabola(x, speed, angle, THROWER_HEIGHT);
                if (y < blockerHeight || circleCrossDot(x, y, BALL_RADIUS, distanceToBlocker, blockerHeight)) return false;
            }
            for (double x = hoopX1 - BALL_RADIUS * 2; x < hoopX2; x = x + 0.01)
            {
                double y = parabola(x, speed, angle, THROWER_HEIGHT);
                if (y < 2.5) return false;
                if (circlesCross(x, y, BALL_RADIUS, hoopX1, HOOP_HEIGHT, HOOP_BAR_RADIUS)) return false;
                else if (circlesCross(x, y, BALL_RADIUS, hoopX2, HOOP_HEIGHT, HOOP_BAR_RADIUS)) return false;
                else if (goodPos(x, y, goodPosXMin, goodPosXMax, VERTICAL_BALL_MIN, VERTICAL_BALL_MAX, precise))
                {
                    return true;
                }
            }
            return false;
        }
    }
}
