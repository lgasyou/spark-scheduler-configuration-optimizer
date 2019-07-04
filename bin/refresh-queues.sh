#!/usr/bin/env bash

hadoop_home=$1

${hadoop_home}/bin/yarn rmadmin -refreshQueues
