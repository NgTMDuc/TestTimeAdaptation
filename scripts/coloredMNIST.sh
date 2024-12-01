GPU=1
OUTPUT="../output/coloredMNIST_new/"
SEED=2024
DSET="ColoredMNIST"
LRMUL=5
INTERVAL=10
BATCH=64
MODEL=resnet18_bn
EXP_TYPE=spurious

#EATA
EATA_FISHERS=1
FISHER_SIZE=2000
FISHER_ALPHA=2000
E_MARGIN=0.4
D_MARGIN=0.5

#SAR
SAR_MARGIN_E0=0.4
IMBALANCE_RATIO=500000

#DEYO
AUG_TYPE=patch
DEYO_MARGIN=0.5
DEYO_MARGIN_0=1
THRSH=0.5

FILTER_ENT=1
FILTER_PLPD=1
REWEIGHT_ENT=1
REWEIGHT_PLPD=1


# python3 ../main.py --exp_type $EXP_TYPE --method no_adapt --dset $DSET --wandb_interval $INTERVAL --model $MODEL --seed $SEED  --gpu $GPU --output $OUTPUT --lr_mul $LRMUL

# python3 ../main.py --exp_type $EXP_TYPE --method tent --dset $DSET --wandb_interval $INTERVAL --model $MODEL --seed $SEED  --gpu $GPU --output $OUTPUT --lr_mul $LRMUL

# python3 ../main.py --exp_type $EXP_TYPE --method eata --dset $DSET --wandb_interval $INTERVAL --model $MODEL --seed $SEED --gpu $GPU --output $OUTPUT --fisher_alpha $FISHER_ALPHA --e_margin $E_MARGIN --d_margin $D_MARGIN --fisher_size $FISHER_SIZE --eata_fishers $EATA_FISHERS --lr_mul $LRMUL

# python3 ../main.py --exp_type $EXP_TYPE --method sar --dset $DSET --wandb_interval $INTERVAL --model $MODEL --seed $SEED --gpu $GPU --output $OUTPUT --sar_margin_e0  $SAR_MARGIN_E0 --imbalance_ratio $IMBALANCE_RATIO --lr_mul $LRMUL

# python3 ../main.py --exp_type $EXP_TYPE --method deyo --dset $DSET --wandb_interval $INTERVAL --model $MODEL --seed $SEED --gpu $GPU --output $OUTPUT --deyo_margin $DEYO_MARGIN --deyo_margin_e0 $DEYO_MARGIN_0 --plpd_threshold $THRSH --filter_ent 0 --filter_plpd $FILTER_PLPD --reweight_ent $REWEIGHT_ENT --reweight_plpd $REWEIGHT_PLPD  --lr_mul $LRMUL

for i in $(seq 1 1 4)
do 
    # for alpha in $(seq 0.1 0.1 0.9)
    # do
    # echo i
    python3 ../main.py --exp_type $EXP_TYPE --method propose --dset $DSET --wandb_interval $INTERVAL --model $MODEL --seed $SEED --gpu $GPU --output $OUTPUT --deyo_margin $DEYO_MARGIN --deyo_margin_e0 $DEYO_MARGIN_0 --plpd_threshold $THRSH --filter_ent $FILTER_ENT --filter_plpd $FILTER_PLPD --reweight_ent $REWEIGHT_ENT --reweight_plpd $REWEIGHT_PLPD  --lr_mul $LRMUL --layer $i
    # done
done