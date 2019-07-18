#!/usr/bin/env bash

hadoop_home=$1

${hadoop_home}/bin/hdfs dfs -rm -r /user/lzq/.sparkStaging/*
