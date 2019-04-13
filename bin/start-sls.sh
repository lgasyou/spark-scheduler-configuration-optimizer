#!/usr/bin/env bash

hadoop_home=$1
work_dir=$2
sls_input_jobs_path=$3
sls_home=${hadoop_home}/share/hadoop/tools/sls

export PATH=${hadoop_home}/bin:$PATH

rm -rf ${work_dir}/results/logs/
mkdir  ${work_dir}/results/logs/
rm -rf ${sls_home}/yumh-err.txt

nodes=${work_dir}/data/testset/sls-nodes.json
output_dir=${work_dir}/results/logs/

export HADOOP_ROOT_LOGGER=ERROR,console
${sls_home}/bin/slsrun.sh --input-sls=${sls_input_jobs_path} --nodes=${nodes} --output-dir=${output_dir} --print-simulation
