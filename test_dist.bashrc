#!/bin/bash

for args in `seq 1 100`;
do
qsub test_dist.PBS -v "args=$args"
done
