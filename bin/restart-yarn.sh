#!/usr/bin/env bash

hadoop_sbin=$1/sbin

"${hadoop_sbin}"/stop-yarn.sh
"${hadoop_sbin}"/start-yarn.sh
