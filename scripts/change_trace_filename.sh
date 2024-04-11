#!/bin/bash -l

if [ $# -eq 0 ]
then
    >&2 echo "[ERROR] The trace directory must be provided"
    exit 1
fi

trace_dir="$1"
mv ${trace_dir}pmpi-trace-rank-00.txt ${trace_dir}pmpi-trace-rank-0.txt
mv ${trace_dir}pmpi-trace-rank-01.txt ${trace_dir}pmpi-trace-rank-1.txt
mv ${trace_dir}pmpi-trace-rank-02.txt ${trace_dir}pmpi-trace-rank-2.txt
mv ${trace_dir}pmpi-trace-rank-03.txt ${trace_dir}pmpi-trace-rank-3.txt
mv ${trace_dir}pmpi-trace-rank-04.txt ${trace_dir}pmpi-trace-rank-4.txt
mv ${trace_dir}pmpi-trace-rank-05.txt ${trace_dir}pmpi-trace-rank-5.txt
mv ${trace_dir}pmpi-trace-rank-06.txt ${trace_dir}pmpi-trace-rank-6.txt
mv ${trace_dir}pmpi-trace-rank-07.txt ${trace_dir}pmpi-trace-rank-7.txt
mv ${trace_dir}pmpi-trace-rank-08.txt ${trace_dir}pmpi-trace-rank-8.txt
mv ${trace_dir}pmpi-trace-rank-09.txt ${trace_dir}pmpi-trace-rank-9.txt
