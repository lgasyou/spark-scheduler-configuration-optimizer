#!/usr/bin/env bash

CLASS=$1
export SPARK_HOME=$2
export HADOOP_HOME=$3
export HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop
export JAVA_HOME=$4
QUEUE=$5
DATA_SIZE=$6
WORK_DIR=$7
JAR_DIR="${WORK_DIR}"/data/workloads
MNIST_DIR="${WORK_DIR}"/data/mnist
OUTPUT_DIR="${WORK_DIR}"/results/workload-out

echo "Starting workload with parameters: class ${CLASS}, queue ${QUEUE}, data_size ${DATA_SIZE}"
case ${CLASS} in
"rnn")
    "${SPARK_HOME}"/bin/spark-submit \
        --class com.intel.analytics.bigdl.models.rnn.workload_rnn \
        --master yarn \
        --deploy-mode cluster \
        --executor-cores 2 \
        --num-executors 2 \
        --executor-memory 6g \
        --driver-memory 2g \
        --queue "${QUEUE}" \
        "${JAR_DIR}" \
        -b 8 \
        -f "${MNIST_DIR}" \
        -s "${OUTPUT_DIR}" \
        --checkpoint "${OUTPUT_DIR}" \
        -e "${DATA_SIZE}"

     rm "${OUTPUT_DIR}"/*
     ;;

"autoencoder" | "lenet")
    "${SPARK_HOME}"/bin/spark-submit \
        --class com.intel.analytics.bigdl.models."${CLASS}".workload_"${CLASS}" \
        --master yarn \
        --deploy-mode cluster \
        --executor-cores 2 \
        --num-executors 2 \
        --executor-memory 6g \
        --driver-memory 2g \
        --queue "${QUEUE}" \
        "${JAR_DIR}" \
        -b 8 \
        -f "${MNIST_DIR}"/cifar-10/ \
        -e "${DATA_SIZE}"
    rm "${OUTPUT_DIR}"/*
    ;;

"resnet" | "vgg")
    "${SPARK_HOME}"/bin/spark-submit \
        --class com.intel.analytics.bigdl.models."${CLASS}".workload_"${CLASS}" \
        --master yarn \
        --deploy-mode cluster \
        --executor-cores 2 \
        --num-executors 2 \
        --executor-memory 6g \
        --driver-memory 2g \
        --queue "${QUEUE}" \
        "${JAR_DIR}" \
        -b 8 \
        -f "${MNIST_DIR}" \
        -e "${DATA_SIZE}"

    rm "${OUTPUT_DIR}"/*
    ;;

*)
    spark-submit \
        --class "${CLASS}" \
        --master yarn \
        --deploy-mode cluster \
        --executor-cores 2 \
        --num-executors 2 \
        --executor-memory 6g \
        --driver-memory 2g \
        "${JAR_DIR} ${DATA_SIZE}"
    ;;
esac
