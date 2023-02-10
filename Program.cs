using System;
using System.Collections.Generic;

namespace Assignment4
{
    class Program
    {
        static string mainFolderPath;
        static string mode;
        static Helper hp = new Helper();
        static FeatureExtraction fe = new FeatureExtraction();
        static Models models = new Models();
        private static Evaluation ev = new Evaluation();

        static void Main(string[] args)
        {
            mainFolderPath = args[1];
            mode = args[3];
            string[] techniques = { "FCTH", "CEDD", "SURF" };
            if (mode.Equals("precompute"))
            {
                List<string>[] pathList = hp.readDataset(mainFolderPath);
                List<string>[] labels = hp.getClassLabels(pathList);

                foreach (string technique in techniques)
                {
                    fe.extractFeatures(pathList, labels, technique);
                }
            }
            else if (mode.Equals("trainval"))
            {
                foreach (string technique in techniques)
                {
                    string trainPath = $"./precomputed_{technique}_train.csv";
                    string valPath = $"./precomputed_{technique}_val.csv";

                    var result = hp.ReadCSV(trainPath);
                    double[][] trainValues = result.Item1;
                    int[] trainLabels = result.Item2;

                    result = hp.ReadCSV(valPath);
                    double[][] valValues = result.Item1;
                    int[] valLabels = result.Item2;

                    var output = hp.deleteUnnecessaryColumns(trainValues, valValues);
                    trainValues = output.Item1;
                    valValues = output.Item2;


                    ev.evaluation(trainValues, trainLabels, valValues, valLabels, technique);
                }
            }

            Console.ReadLine();
        }
    }
}