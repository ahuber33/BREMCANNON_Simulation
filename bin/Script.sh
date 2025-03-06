#!/bin/bash


echo "Début de traitement"
#myArr=("1" "2" "5" "7" "10" "25" "50" "75" "100" "200" "300" "400" "500" "600" "700" "800" "900" "1000" "2000")
myArr=("1000" "750" "500" "400" "300" "250" "200" "150" "100" "90" "80" "70" "60" "50" "40" "30" "20" "10" "8" "6" "4" "2" "1")
#myArr=("1000")
div=1000
string1="Exp_"
string2="MeV_10M"

#for idx in ${!myArr[*]}
for ((idx=74; idx<=100; idx+=1))
do
    while [[ $(pgrep -x CEDRICLULI2023S | wc -l) -gt 7 ]]
    do
	sleep 10
    done
        #Init
    #value=${myArr[$idx]}
    value=$(bc <<< "scale=3 ; $idx/$div")
    #value=$idx
    #n=1
    #value1=$(bc <<< "scale=1 ; $idx+1")
    #echo $value1
    #name="${string1}${value}${string2}" 
    #echo $name
    idx_fichier=$(($idx))
    echo Index : $idx / Fichier : $value
        sleep 2
        # Creation du fichier a partir du fichier de base (fichier temporaire)
        cp vrml_base.mac base_$idx_fichier_bis.mac
        # Mise en place de la variable
        #sed -e "s/%energy1/$value/g" base_$idx_fichier_bis.mac > base_bis_$idx_fichier.mac
	#sed -e "s/%energy2/$value1/g" base_bis_$idx_fichier.mac > base_$idx_fichier.mac
        sed -e "s/%energy/$value/g" base_$idx_fichier_bis.mac > base_$idx_fichier.mac
        # Suppression fichier temporaire
        rm base_$idx_fichier_bis.mac
	rm base_bis_$idx_fichier.mac
	#./ENLCEDRICSim Electron_Uniform_${value}_MeV_1M 1000000 base_$idx_fichier.mac &
	./CEDRICLULI2023Sim Exp_${idx}keV_1M 1000000 base_$idx_fichier.mac &
	#./CEDRICLULI2023Sim Config_19_${value}keV_10M 10000000 base_$idx_fichier.mac &
	#./ENLCEDRICSim $name 10000000 base_$idx_fichier.mac &
	#./ENLCEDRICSim Exp_${myArr[$idx]}MeV_100M 100000000 vrml.mac &
	sleep 5
done
echo "Fin de traitement"


