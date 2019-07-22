from optimizer.environment.timedelayprediction.sparkworkloadrandomgenerator import SparkWorkloadRandomGenerator
from workloadsubmission.controlmodel import workload_interface


gen = SparkWorkloadRandomGenerator()
workloads = gen.generate()
print(workloads)
workload_interface(workloads)
