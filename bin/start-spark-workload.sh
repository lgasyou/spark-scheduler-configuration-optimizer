#!/usr/bin/env bash

spark_home=$2
hadoop_conf_dir=$3/etc/hadoop
java_home=$4
work_dir=$5
queue=$6
size=$7

export JAVA_HOME=${java_home}
export HADOOP_CONF_DIR=${hadoop_conf_dir}


function submit() {
    class=$1
    queue=$2
    workload=$3
    size=$4

    ${spark_home}/bin/spark-submit \
    --class ${class} \
    --master yarn \
    --deploy-mode cluster \
    --queue ${queue} \
    --num-executors 5 \
    --executor-cores 4 \
    --executor-memory 6g \
    --driver-memory 1g \
    ${work_dir}/data/testset/${workload}.jar ${size}
}

case $1 in
    SVM)
        submit svm ${queue} workload_SVM ${size}
        ;;

    fpgrowth)
        submit FPGrowth ${queue} workload_FPGrowth ${size}
        ;;

    kmeans)
        submit kmeans ${queue} workload_kmeans ${size}
        ;;

    linear)
        submit linear ${queue} workload_linear ${size}
        ;;

    lda)
        submit lda ${queue} workload_lda ${size}
        ;;

    bayes)
        submit bayes ${queue} workload_bayes ${size}
        ;;

    als)
        submit als ${queue} workload_als ${size}
        ;;
esac
