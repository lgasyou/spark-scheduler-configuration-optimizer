#!/usr/bin/env bash

hadoop_home=$1
work_dir=$2
sls_home=${hadoop_home}/share/hadoop/tools/sls

export PATH=${hadoop_home}/bin:$PATH

rm -rf ${work_dir}/results/logs/
mkdir  ${work_dir}/results/logs/
rm -rf ${sls_home}/yumh-err.txt

input_sls=${work_dir}/data/testset/sls-jobs.json
nodes=${work_dir}/data/testset/sls-nodes.json
output_dir=${work_dir}/results/logs/

${sls_home}/bin/slsrun.sh --input-sls=${input_sls} --nodes=${nodes} --output-dir=${output_dir} --print-simulation
