DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# declare -a tasks=('FactoryTaskGears' 'FactoryTaskInsertion' 'FactoryTaskNutBoltPick' 'FactoryTaskNutBoltScrew')
declare -a tasks=('FactoryTaskInsertion')

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py task=${tasks[$i]} headless=True wandb_activate=True wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
done


# python train.py task=FactoryTaskInsertion headless=False test=True checkpoint='runs/FactoryTaskInsertion/nn/last_FactoryTaskInsertionep8192rew\[-212.67\].pth'
# python train.py task=FactoryTaskInsertion headless=False test=True checkpoint='runs/FactoryTaskInsertion/nn/FactoryTaskInsertion.pth'