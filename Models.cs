using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;

namespace Assignment4
{
    class Models
    {
        public Models()
        {
        }

        public RandomForest randomForest(double[][] trainValues, int[] labelValues)
        {
            // Create a new Random Forest for classification
            RandomForestLearning forestLearning = new RandomForestLearning(){NumberOfTrees = 25};

            var model = forestLearning.Learn(trainValues, labelValues);
        
            return model;
        }

        public int[] svm(double[][] trainValues, int[] labelValues, double[][] valValues)
        {
            // Create a one-vs-one multi-class SVM learning algorithm 
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // The following line is only needed to ensure reproducible results. Please remove it to enable full parallelization
            //teacher.ParallelOptions.MaxDegreeOfParallelism = 1;

            // Learn a machine
            var model = teacher.Learn(trainValues, labelValues);
            int[] predicted = model.Decide(valValues);
            return predicted;
        }

        public DecisionTree c45(double[][] trainValues, int[] labelValues)
        {
            C45Learning teacher = new C45Learning();

            // Finally induce the tree from the data:
            var model = teacher.Learn(trainValues, labelValues);


            return model;
        }
    }
}