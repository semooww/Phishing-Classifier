using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Printing;
using Accord.Imaging;
using Accord.MachineLearning;
using Accord.Math;
using Accord.Statistics.Filters;
using CEDD_Descriptor;
using FCTH_Descriptor;

namespace Assignment4
{
    class FeatureExtraction
    {
        private KMeansClusterCollection clusters;
        Helper hp = new Helper();
        int numClusters = 400; // Number of clusters

        public FeatureExtraction()
        {
        }

        public void extractFeatures(List<string>[] pathList, List<string>[] labels, string technique)
        {
            string data = null;

            for (int i = 0; i < pathList.Length; i++) // Train images - Val images
            {
                if (i == 0)
                {
                    data = "train";
                }
                else if (i == 1)
                {
                    data = "val";
                }

                Console.WriteLine(technique + " features are being extracted for " + data + "...");
                var watch = new Stopwatch();
                watch.Start();
                List<double[]> features = new List<double[]>();
                double[][][] surfFeatureArray = new double[pathList[i].Count][][];
                for (int j = 0; j < pathList[i].Count; j++)
                {
                    Bitmap img = new Bitmap(pathList[i][j]);
                    img = new Bitmap(img, new Size(1024, 1024));

                    if (technique.Equals("CEDD"))
                    {
                        CEDD_Descriptor.CEDD cedd = new CEDD_Descriptor.CEDD();
                        double[] ceddTable = cedd.Apply(img);
                        features.Add(ceddTable);
                    }
                    else if (technique.Equals("FCTH"))
                    {
                        FCTH_Descriptor.FCTH fcth = new FCTH_Descriptor.FCTH();
                        double[] fcthTable = fcth.Apply(img, 2);
                        features.Add(fcthTable);
                    }
                    else if (technique.Equals("SURF"))
                    {
                        SpeededUpRobustFeaturesDetector surf = new SpeededUpRobustFeaturesDetector();
                        // Use it to extract the SURF point descriptors from the image:

                        List<SpeededUpRobustFeaturePoint> descriptors = surf.ProcessImage(img);

                        // We can obtain the actual double[][] descriptors using
                        double[][] featureArray = descriptors.Apply(d => d.Descriptor);
                        surfFeatureArray[j] = featureArray;
                    }
                }

                if (technique.Equals("SURF"))
                {
                    features = kmeans(surfFeatureArray, data);
                    double[][] result = rowNormalization(features.ToArray());
                    hp.formatCSV(result, labels[i], technique, data);
                }
                else
                {
                    hp.formatCSV(features.ToArray(), labels[i], technique, data);
                }

                watch.Stop();
                Console.Write($"{watch.ElapsedMilliseconds / 1000} seconds\n");
            }
        }


        public List<double[]> kmeans(double[][][] surfFeatures, string data)
        {
            if (data.Equals("train"))
            {
                double[][] allFeatures = hp.convert3dto2d(surfFeatures);
                KMeans kmeans = new KMeans(k: numClusters);
                // Compute and retrieve the data centroids
                this.clusters = kmeans.Learn(allFeatures);
            }

            List<double[]> features = new List<double[]>();
            for (int i = 0; i < surfFeatures.Length; i++)
            {
                int[] kmeansOutput = this.clusters.Decide(surfFeatures[i]);
                double[] extracted = quantizeLocalDescriptorOutput(kmeansOutput);
                features.Add(extracted);
            }

            return features;
        }

        public double[] quantizeLocalDescriptorOutput(int[] kMeansOutput)
        {
            double[] output = new double[numClusters];

            for (int i = 0; i < kMeansOutput.Length; i++)
            {
                output[kMeansOutput[i]] = output[kMeansOutput[i]] + 1;
            }

            return output;
        }


        public double[][] rowNormalization(double[][] features)
        {
            double[][] normalized = new double[features.Length][];
            for (int i = 0; i < features.Length; i++)
            {
                double total = 0;
                double[] row = new double[features[i].Length];
                for (int k = 0; k < 2; k++)
                {
                    for (int j = 0; j < features[i].Length; j++)
                    {
                        if (k == 0)
                        {
                            total += features[i][j];
                        }
                        else
                        {
                            row[j] = features[i][j] / total;
                        }
                    }
                }

                normalized[i] = row;
            }

            return normalized;
        }

        public double[][] colNormalization(double[][] features)
        {
            double[] totals = new double[features[0].Length];
            for (int j = 0; j < features[0].Length; j++)
            {
                for (int i = 0; i < features.Length; i++)
                {
                    totals[j] += features[i][j];
                }
            }

            double[][] normalized = new double[features.Length][];
            for (int i = 0; i < features.Length; i++)
            {
                double[] row = new double[features[i].Length];
                for (int j = 0; j < features[i].Length; j++)
                {
                    row[j] = features[i][j] / totals[j];
                }

                normalized[i] = row;
            }

            return normalized;
        }
    }
}