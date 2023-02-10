using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Accord.MachineLearning.DecisionTrees;
using static System.Globalization.CultureInfo;

namespace Assignment4
{
    class Helper
    {
        public Helper()
        {
        }

        public List<string>[] readDataset(string folderPath)
        {
            Console.WriteLine("Reading phishIRIS_DL_Dataset...");
            List<string> pathList_forTrain = new List<string>();
            List<string> pathList_forVal = new List<string>();
            string[] folderArray = Directory.GetDirectories(folderPath);

            int classCounter = 0; // Count how many class we have in the dataset
            int fileCounter = 0; // Count how many files we have in the dataset
            for (int i = 0; i < folderArray.Length; i++)
            {
                classCounter = 0; //reset class counter
                fileCounter = 0; //reset file counter

                string folderName = folderArray[i].Split(Path.DirectorySeparatorChar)[2];
                string[] subFolderArray = Directory.GetDirectories(folderArray[i]);
                for (int j = 0; j < subFolderArray.Length; j++)
                {
                    string subFolderName = subFolderArray[j].Split(Path.DirectorySeparatorChar)[3];
                    classCounter++;
                    string[] filePaths = Directory.GetFiles(subFolderArray[j], "*");
                    for (int k = 0; k < filePaths.Length; k++)
                    {
                        string filePath = filePaths[k];
                        if (i == 0)
                        {
                            pathList_forTrain.Add(filePath);
                        }
                        else if (i == 1)
                        {
                            pathList_forVal.Add(filePath);
                        }

                        fileCounter++;
                    }
                }

                Console.WriteLine(fileCounter.ToString() + " images were found in " +
                                  folderArray[i].Split(Path.DirectorySeparatorChar)[2] + " folder");
            }

            List<string>[] paths = { pathList_forTrain, pathList_forVal };

            Console.WriteLine((classCounter - 1).ToString() + " classes exist");
            return paths;
        }

        public List<string>[] getClassLabels(List<string>[] paths)
        {
            List<string> classLabelsforTrain = new List<string>();
            List<string> classLabelsforVal = new List<string>();
            int i = 0;
            foreach (List<string> mode in paths)
            {
                foreach (string path in mode)
                {
                    string label = path.Split(Path.DirectorySeparatorChar)[3];
                    if (i == 0)
                    {
                        classLabelsforTrain.Add(label);
                    }
                    else if (i == 1)
                    {
                        classLabelsforVal.Add(label);
                    }
                }

                i++;
            }

            List<string>[] classLabels = { classLabelsforTrain, classLabelsforVal };
            return classLabels;
        }

        public int convertStringLabeltoInteger(string label)
        {
            switch (label)
            {
                case "adobe":
                    return 0;
                case "alibaba":
                    return 1;
                case "amazon":
                    return 2;
                case "apple":
                    return 3;
                case "boa":
                    return 4;
                case "chase":
                    return 5;
                case "dhl":
                    return 6;
                case "dropbox":
                    return 7;
                case "facebook":
                    return 8;
                case "linkedin":
                    return 9;
                case "microsoft":
                    return 10;
                case "other":
                    return 11;
                case "paypal":
                    return 12;
                case "wellsfargo":
                    return 13;
                case "yahoo":
                    return 14;
            }

            return -1;
        }

        public void formatCSV(double[][] features, List<string> labels, string technique, string whichData)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < features[0].Length; i++)
            {
                string f = "f" + (i + 1) + ",";
                sb.Append(f);
            }

            sb.Append("label");
            sb.AppendLine();
            int labelCounter = 0;
            foreach (double[] array in features)
            {
                for (int i = 0; i < array.Length; i++)
                {
                    string element = array[i].ToString("0.00000", InvariantCulture);
                    sb.Append(element + ",");
                }

                sb.Append(labels[labelCounter++]);
                sb.AppendLine();
            }

            string currentDir = Directory.GetCurrentDirectory();
            string fileName = "precomputed_" + technique + "_" + whichData + ".csv";

            using (StreamWriter theWriter = new StreamWriter(currentDir + "\\" + fileName))
            {
                theWriter.Write(sb.ToString().Trim());
            }

            Console.Write("Done. " + fileName + " is regenerated in ");
        }

        public (double[][], int[]) ReadCSV(string filePath)
        {
            // Read the file into an array of lines
            string[] lines = File.ReadAllLines(filePath);

            List<double[]> output = new List<double[]>();
            List<int> outputLabel = new List<int>();
            // Iterate over the lines
            bool firstLine = true;
            foreach (string line in lines)
            {
                if (firstLine)
                {
                    firstLine = false;
                    continue;
                }

                // Split the line into an array of fields
                string[] fields = line.Split(',');
                List<double> outputLine = new List<double>();
                for (int i = 0; i < fields.Length; i++)
                {
                    if (i == (fields.Length - 1))
                    {
                        outputLabel.Add(convertStringLabeltoInteger(fields[i]));
                    }
                    else
                    {
                        outputLine.Add(Double.Parse(fields[i]));
                    }
                }

                output.Add(outputLine.ToArray());
                // Print the fields
                // Console.WriteLine(string.Join(", ", fields));
            }

            return (output.ToArray(), outputLabel.ToArray());
        }

        public double[][] convert3dto2d(double[][][] features)
        {
            int rowCount = 0;
            for (int i = 0; i < features.Length; i++)
            {
                rowCount += features[i].Length;
            }

            double[][] allFeatures = new double[rowCount][];
            int index = 0;
            for (int i = 0; i < features.Length; i++)
            {
                for (int j = 0; j < features[i].Length; j++)
                {
                    allFeatures[index++] = features[i][j];
                }
            }

            return allFeatures;
        }

        public int[] findIndexes(double[][] array)
        {
            List<int> indexes = new List<int>();
            IList<DecisionVariable> att = (IList<DecisionVariable>)DecisionVariable.FromData(array);
            for (int i = 0; i < array[0].Length; i++)
            {
                if (att[i].Range.Length == 0)
                {
                    indexes.Add(i);
                }
            }

            return indexes.ToArray();
        }

        public (double[][], double[][]) deleteUnnecessaryColumns(double[][] trainValues, double[][] valValues)
        {
            int[] indexes = findIndexes(trainValues);
            List<double[]> resTrainArray = new List<double[]>();
            List<double[]> resValArray = new List<double[]>();
            for (int i = 0; i < trainValues.Length; i++)
            {
                List<double> row = new List<double>();
                for (int j = 0; j < trainValues[0].Length; j++)
                {
                    if (Array.Exists(indexes, element => element == j))
                    {
                        continue;
                    }
                    else
                    {
                        row.Add(trainValues[i][j]);
                    }
                }

                resTrainArray.Add(row.ToArray());
            }

            for (int i = 0; i < valValues.Length; i++)
            {
                List<double> row = new List<double>();
                for (int j = 0; j < valValues[0].Length; j++)
                {
                    if (Array.Exists(indexes, element => element == j))
                    {
                        continue;
                    }
                    else
                    {
                        row.Add(valValues[i][j]);
                    }
                }

                resValArray.Add(row.ToArray());
            }

            return (resTrainArray.ToArray(), resValArray.ToArray());
        }
    }
}