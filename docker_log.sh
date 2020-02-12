#!/bin/sh
logs=$(find /var/lib/docker/containers/ -name *-json.log)

for log in $logs
    do
        ls -lh $log
        head -100 $log
    done