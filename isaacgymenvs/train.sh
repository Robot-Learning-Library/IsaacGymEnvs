DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

declare -a tasks=('FactoryTaskGears' 'FactoryTaskInsertion' 'FactoryTaskNutBoltPick' 'FactoryTaskNutBoltScrew')

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py task=${tasks[$i]} wandb_activate=True wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
done
