#!/usr/bin/env bash

hadoop_sbin=/home/lzq/library/hadoop/sbin

${hadoop_sbin}/stop-yarn.sh
${hadoop_sbin}/start-yarn.sh
