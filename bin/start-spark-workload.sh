#!/usr/bin/env bash

CLASS=$1
export SPARK_HOME=$2
export HADOOP_HOME=$3
export HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop
export JAVA_HOME=$4
QUEUE=$5
DATA_SIZE=$6
PROJECT_DIR=$7
WORKLOAD_JAR_DIR="${PROJECT_DIR}"/data/workloads
MNIST_DIR="${PROJECT_DIR}"/data/mnist
OUTPUT_DIR="${PROJECT_DIR}"/results/workload-out

EXECUTOR_CORES=2
NUM_EXECUTORS=2
EXECUTOR_MEMORY=6g
DRIVER_MEMORY=2g

echo "Starting workload with parameters: class ${CLASS}, queue ${QUEUE}, data_size ${DATA_SIZE}"
case ${CLASS} in
"rnn")
    spark-submit \
        --class com.intel.analytics.bigdl.models.rnn.workload_rnn \
        --master yarn \
        --deploy-mode cluster \
        --executor-cores "${EXECUTOR_CORES}" \
        --num-executors "${NUM_EXECUTORS}" \
        --executor-memory "${EXECUTOR_MEMORY}" \
        --driver-memory "${DRIVER_MEMORY}" \
        --queue "${QUEUE}" \
        "${WORKLOAD_JAR_DIR}"/workload_rnn.jar \
        -b 8 \
        -f "${MNIST_DIR}" \
        -s "${OUTPUT_DIR}" \
        --checkpoint "${OUTPUT_DIR}" \
        -e "${DATA_SIZE}"

    rm -r "${OUTPUT_DIR:?}"/*
    ;;

"autoencoder" | "lenet")
    spark-submit \
        --class com.intel.analytics.bigdl.models."${CLASS}".workload_"${CLASS}" \
        --master yarn \
        --deploy-mode cluster \
        --executor-cores "${EXECUTOR_CORES}" \
        --num-executors "${NUM_EXECUTORS}" \
        --executor-memory "${EXECUTOR_MEMORY}" \
        --driver-memory "${DRIVER_MEMORY}" \
        --queue "${QUEUE}" \
        "${WORKLOAD_JAR_DIR}/workload_${CLASS}.jar" \
        -b 8 \
        -f "${MNIST_DIR}" \
        -e "${DATA_SIZE}"
    rm -r "${OUTPUT_DIR:?}"/*
    ;;

"resnet" | "vgg")
    spark-submit \
        --class com.intel.analytics.bigdl.models."${CLASS}".workload_"${CLASS}" \
        --master yarn \
        --deploy-mode cluster \
        --executor-cores "${EXECUTOR_CORES}" \
        --num-executors "${NUM_EXECUTORS}" \
        --executor-memory "${EXECUTOR_MEMORY}" \
        --driver-memory "${DRIVER_MEMORY}" \
        --queue "${QUEUE}" \
        "${WORKLOAD_JAR_DIR}/workload_${CLASS}.jar" \
        -b 8 \
        -f "${MNIST_DIR}"/cifar-10 \
        -e "${DATA_SIZE}"

    rm -r "${OUTPUT_DIR:?}"/*
    ;;

*)
    spark-submit \
        --class "${CLASS}" \
        --master yarn \
        --deploy-mode cluster \
        --executor-cores "${EXECUTOR_CORES}" \
        --num-executors "${NUM_EXECUTORS}" \
        --executor-memory "${EXECUTOR_MEMORY}" \
        --driver-memory "${DRIVER_MEMORY}" \
        --queue "${QUEUE}" \
        "${WORKLOAD_JAR_DIR}/workload_${CLASS}.jar" "${DATA_SIZE}"
    ;;
esac
