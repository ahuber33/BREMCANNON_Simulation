 #!/bin/bash


for (( i=1 ; i<=8 ; i++ ))
do
    while [[ $(pgrep -x ENLCEDRICSim | wc -l) -gt 3 ]]
    do
	sleep 10
    done
    ./ENLCEDRICSim a_${i} 1250000 vrml.mac &
    sleep 5
    done
