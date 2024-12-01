DSET="ColoredMNIST"
METHOD="propose"
LAYER=2
TRANSFORM="AdaIN"
EXP="spurious"

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
DEYO_MARGIN_0=0.4
THRSH=0.5

FILTER_ENT=1
FILTER_PLPD=1
REWEIGHT_ENT=1
REWEIGHT_PLPD=1
LRMUL=5
INTERVAL=10
# GPU=1,2
# for i in {1..4}
# do 
#     python3 ../main.py --dset $DSET --method $METHOD --layer $LAYER --transform $TRANSFORM --exp_type $EXP --layer $i --wandb_interval $INTERVAL --deyo_margin $DEYO_MARGIN --deyo_margin_e0 $DEYO_MARGIN_0 --plpd_threshold $THRSH --filter_ent $FILTER_ENT --filter_plpd $FILTER_PLPD --reweight_ent $REWEIGHT_ENT --reweight_plpd $REWEIGHT_PLPD  --lr_mul $LRMUL
# done
METHOD=deyo
python3 ../main.py --dset $DSET --method $METHOD --layer $LAYER --transform $TRANSFORM --exp_type $EXP --wandb_interval $INTERVAL --deyo_margin $DEYO_MARGIN --deyo_margin_e0 $DEYO_MARGIN_0 --plpd_threshold $THRSH --filter_ent $FILTER_ENT --filter_plpd $FILTER_PLPD --reweight_ent $REWEIGHT_ENT --reweight_plpd $REWEIGHT_PLPD  --lr_mul $LRMUL

