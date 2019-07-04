#!/usr/bin/env bash

hadoop fs -put ../data/testset/sample_libsvm_data.txt /ml
hadoop fs -ls -h /ml
