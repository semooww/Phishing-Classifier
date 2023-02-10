using System;
using System.Collections.Generic;
using System.Diagnostics;
using Accord.MachineLearning.DecisionTrees;
using Accord.Statistics.Analysis;
using static System.Globalization.CultureInfo;

namespace Assignment4
{
    public class Evaluation
    {
        private Models models = new Models();

        public Evaluation()
        {
        }


        public void evaluation(double[][] trainValues, int[] trainLabels, double[][] valValues, int[] valLabels, string technique)
        {
            var watch = new Stopwatch();
            watch.Start();
            Console.WriteLine($"Training with precomputed_{technique}_train.csv");

            RandomForest rfModel = models.randomForest(trainValues, trainLabels);
            int[] predicted = models.svm(trainValues, trainLabels, valValues);
            DecisionTree c45Model = models.c45(trainValues, trainLabels);
            watch.Stop();
            Console.WriteLine($"Done in {watch.ElapsedMilliseconds / 1000} seconds");
            Console.WriteLine($"Testing with precomputed_{technique}_val.csv {valValues.Length} samples");
            testRF(rfModel, valValues, valLabels);
            testSVM(predicted, valLabels);
            testC45(c45Model, valValues, valLabels);
            Console.WriteLine("---------------------------------------------------");
        }

        public void testRF(RandomForest rfModel, double[][] valValues, int[] valLabels)
        {
            int[] predicted = rfModel.Decide(valValues);
            var metrics = confusionMatrix(predicted, valLabels);
            print(metrics, "Random Forest");
        }

        public void testSVM(int[] predicted, int[] valLabels)
        {
            var metrics = confusionMatrix(predicted, valLabels);
            print(metrics, "SVM");
        }

        public void testC45(DecisionTree c45Model, double[][] valValues, int[] valLabels)
        {
            int[] predicted = c45Model.Decide(valValues);
            var metrics = confusionMatrix(predicted, valLabels);
            print(metrics, "C45");
        }

        public void print((double, double, double) metrics, string modelName)
        {
            string tpr = metrics.Item1.ToString("0.000", InvariantCulture);
            string fpr = metrics.Item2.ToString("0.000", InvariantCulture);
            string f1 = metrics.Item3.ToString("0.000", InvariantCulture);
            string output = String.Format("{0,-14}| {1,-5}{2:0.000} | {3,-4}{4:0.000} | {5,-4}{6:0.000}", modelName, "TPR", tpr, "FPR", fpr, "F1", f1);
            Console.WriteLine(output);
        }

        public (double, double, double) confusionMatrix(int[] predicted, int[] outputs)
        {
            // Evaluate the predictions
            GeneralConfusionMatrix matrix = new GeneralConfusionMatrix(predicted, outputs);
            // Calculate the TPR, FPR and F-1 score for each class
            int numClasses = matrix.NumberOfClasses;
            int[,] mat = matrix.Matrix;
            return findMetrics(mat);
        }

        public (double, double, double) findMetrics(int[,] mat)
        {
            int numClasses = mat.GetLength(0);
            List<double> tpr = new List<double>();
            List<double> fpr = new List<double>();
            List<double> f1 = new List<double>();
            for (int c = 0; c < numClasses; c++)
            {
                double TP = 0, FP = 0, FN = 0, TN = 0;
                for (int i = 0; i < mat.GetLength(0); i++)
                {
                    for (int j = 0; j < mat.GetLength(1); j++)
                    {
                        if (i == j && c == i)
                        {
                            TP += mat[i, j];
                        }
                        else if (c == i)
                        {
                            FN += mat[i, j];
                        }

                        else if (c == j)
                        {
                            FP += mat[i, j];
                        }
                        else
                        {
                            TN += mat[i, j];
                        }
                    }
                }

                double rateTPR = TP / (TP + FN);
                tpr.Add(rateTPR);
                double rateFPR = FP / (FP + TN);
                fpr.Add(rateFPR);
                double rateF1 = (2 * TP) / (2 * TP + FP + FN);
                f1.Add(rateF1);
            }

            return (getAvarage(tpr), getAvarage(fpr), getAvarage(f1));
        }

        public double getAvarage(List<double> list)
        {
            double total = 0;
            for (int i = 0; i < list.Count; i++)
            {
                total += list[i];
            }

            double result = total / (double)list.Count;
            return result;
        }
    }
}