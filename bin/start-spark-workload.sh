#!/usr/bin/env bash

spark_home=$2
hadoop_conf_dir=$3/etc/hadoop
java_home=$4
work_dir=$5

export JAVA_HOME=${java_home}
export HADOOP_CONF_DIR=${hadoop_conf_dir}


function submit() {
    class=$1
    queue=$2
    workload=$3

    ${spark_home}/bin/spark-submit \
    --class ${class} \
    --master yarn \
    --deploy-mode cluster \
    --queue ${queue} \
    --executor-memory 6g \
    --driver-memory 1g \
    ${work_dir}/data/testset/${workload}.jar
}

case $1 in
    SVM)
        submit scalapackage.runtest queueA workload_SVM
        ;;

    fpgrowth)
        submit scalapackage.FPGrowth queueB workload_fpgrowth
        ;;

    kmeans)
        submit scalapackage.kmeans queueC workload_kmeans
        ;;

    linear)
        submit scalapackage.linear queueD workload_linear
        ;;

    lda)
        submit scalapackage.lda queueA workload_lda
        ;;

    bayes)
        submit scalapackage.bayes queueB workload_bayes
        ;;

    als)
        submit scalapackage.als queueC workload_als
        ;;

esac
