#!/usr/bin/env bash

spark_home=$2
hadoop_conf_dir=$3/etc/hadoop
java_home=$4
work_dir=$5

export JAVA_HOME=${java_home}
export HADOOP_CONF_DIR=${hadoop_conf_dir}


case $1 in
    SVM)
        ${spark_home}/bin/spark-submit \
            --class scalapackage.runtest \
            --master yarn \
            --deploy-mode cluster \
            --queue queueA \
            ${work_dir}/data/testset/workload_SVM.jar
        ;;

    fpgrowth)
        ${spark_home}/bin/spark-submit \
        --class scalapackage.FPGrowth \
        --master yarn \
        --deploy-mode cluster \
        --queue queueB \
        ${work_dir}/data/testset/workload_fpgrowth.jar
        ;;

    kmeans)
        ${spark_home}/bin/spark-submit \
            --class scalapackage.kmeans \
            --master yarn \
            --deploy-mode cluster \
            --queue queueC \
            ${work_dir}/data/testset/workload_kmeans.jar
        ;;

    linear)
        ${spark_home}/bin/spark-submit \
            --class scalapackage.linear \
            --master yarn \
            --deploy-mode cluster \
            --queue queueD \
            ${work_dir}/data/testset/workload_linear.jar
        ;;

    lda)
        ${spark_home}/bin/spark-submit \
            --class scalapackage.lda \
            --master yarn \
            --deploy-mode cluster \
            --queue queueA \
            ${work_dir}/data/testset/workload_lda.jar
        ;;

    bayes)
        ${spark_home}/bin/spark-submit \
            --class scalapackage.bayes \
            --master yarn \
            --deploy-mode cluster \
            --queue queueB \
            ${work_dir}/data/testset/workload_bayes.jar
        ;;
esac
