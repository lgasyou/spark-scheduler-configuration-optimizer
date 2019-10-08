#!/usr/bin/env bash

cd ../results || exit

rm $(ls | grep -v "algorithm-models.pk" | grep -v "README.md" | grep -v "workload-out")
