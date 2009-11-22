#!/bin/bash

command="$1"

unset DISPLAY
nice -n 10 matlab -nodisplay -nojvm > "tmp.matlab.txt" 2>&1 << EOF 
 $command;
 exit 
EOF

rm tmp.matlab.txt 
