#------------------------------EDSR_CVPRW2017---------------------------
#python train.py --opt options/train/EDSR.yaml --name EDSRx2_bs16ps96lr1e-4 --scale 2 --bs 16 --ps 96 --lr 1e-4 --gpu_ids 0 --use_chop
#python train.py --opt options/train/EDSR.yaml --name EDSRx3_bs16ps96lr1e-4 --scale 3 --bs 16 --ps 96 --lr 1e-4 --gpu_ids 0 --use_chop
#python train.py --opt options/train/EDSR.yaml --name EDSRx4_bs16ps96lr1e-4 --scale 4 --bs 16 --ps 96 --lr 1e-4 --gpu_ids 0 --use_chop

#----------------------------IMDN_ACM_MM2019----------------------------
#python train.py --opt options/train/IMDN.yaml --name IMDNx2_bs16ps96lr2e-4 --scale 2 --bs 16 --ps 96 --lr 2e-4 --gpu_ids 1 --use_chop
#python train.py --opt options/train/IMDN.yaml --name IMDNx3_bs16ps96lr2e-4 --scale 3 --bs 16 --ps 96 --lr 2e-4 --gpu_ids 1 --use_chop
#python train.py --opt options/train/IMDN.yaml --name IMDNx4_bs16ps96lr2e-4 --scale 4 --bs 16 --ps 96 --lr 2e-4 --gpu_ids 1 --use_chop

#---------------------------LatticeNet_ECCV2020-------------------------
#python train.py --opt options/train/LatticeNet.yaml --name LatticeNetx2_bs16ps96lr2e-4 --scale 2 --bs 16 --ps 96 --lr 2e-4 --gpu_ids 2 --use_chop
#python train.py --opt options/train/LatticeNet.yaml --name LatticeNetx3_bs16ps96lr2e-4 --scale 3 --bs 16 --ps 96 --lr 2e-4 --gpu_ids 2 --use_chop
#python train.py --opt options/train/LatticeNet.yaml --name LatticeNetx4_bs16ps96lr2e-4 --scale 4 --bs 16 --ps 96 --lr 2e-4 --gpu_ids 2 --use_chop

#-----------------------------CARN_ECCV2018-----------------------------
#python train.py --opt options/train/CARN.yaml --name CARNx2_bs16ps96lr1e-4 --scale 2 --bs 16 --ps 96 --lr 1e-4 --gpu_ids 0 --use_chop
#python train.py --opt options/train/CARN.yaml --name CARNx3_bs16ps96lr1e-4 --scale 3 --bs 16 --ps 96 --lr 1e-4 --gpu_ids 3 --use_chop
#python train.py --opt options/train/CARN.yaml --name CARNx4_bs16ps96lr1e-4 --scale 4 --bs 16 --ps 96 --lr 1e-4 --gpu_ids 1 --use_chop

#-----------------------------IDN_CVPR2018------------------------------
#python train.py --opt options/train/IDN.yaml --name IDNx2_bs16ps96lr2e-4 --scale 2 --bs 16 --ps 96 --lr 2e-4 --gpu_ids 0 --use_chop
#python train.py --opt options/train/IDN.yaml --name IDNx3_bs16ps96lr2e-4 --scale 3 --bs 16 --ps 96 --lr 2e-4 --gpu_ids 1 --use_chop
#python train.py --opt options/train/IDN.yaml --name IDNx4_bs16ps96lr2e-4 --scale 4 --bs 16 --ps 96 --lr 2e-4 --gpu_ids 2 --use_chop

#---------------------------------HFAN----------------------------------
#python train.py --opt options/train/HFAN.yaml --name HFANx2_bs16ps96lr6e-4 --scale 2 --bs 16 --ps 96 --lr 6e-4 --gpu_ids 0 --use_chop
#python train.py --opt options/train/HFAN.yaml --name HFANx3_bs16ps96lr6e-4 --scale 3 --bs 16 --ps 96 --lr 6e-4 --gpu_ids 2 --use_chop
#python train.py --opt options/train/HFAN.yaml --name HFANx4_bs16ps96lr6e-4 --scale 4 --bs 16 --ps 96 --lr 6e-4 --gpu_ids 1 --use_chop


#-------------------------------Ablation--------------------------------
#python train.py --opt options/train/HFAN_ablation_base.yaml --name HFAN_ablation_basex4_bs16ps96lr6e-4 --scale 4 --bs 16 --ps 96 --lr 6e-4 --gpu_ids 0 --use_chop
#python train.py --opt options/train/HFAN_ablation_base_head_tail_conv.yaml --name HFAN_ablation_base_head_tail_convx4_bs16ps96lr6e-4 --scale 4 --bs 16 --ps 96 --lr 6e-4 --gpu_ids 1 --use_chop
#python train.py --opt options/train/HFAN_ablation_base_transformation.yaml --name HFAN_ablation_base_transformationx4_bs16ps96lr6e-4 --scale 4 --bs 16 --ps 96 --lr 6e-4 --gpu_ids 2 --use_chop
#python train.py --opt options/train/HFAN_D4U2.yaml --name HFAN_ablation_allx4_bs16ps96lr6e-4 --scale 4 --bs 16 --ps 96 --lr 6e-4 --gpu_ids 0 --use_chop


#-----------Compare with embedded attention mechanism-------------------
#python train.py --opt options/train/Baseline.yaml --name Baselinex4_bs16ps96lr1e-4 --scale 4 --bs 16 --ps 96 --lr 1e-4 --gpu_ids 0 --use_chop
#python train.py --opt options/train/BaselineCA.yaml --name BaselineCAx4_bs16ps96lr1e-4 --scale 4 --bs 16 --ps 96 --lr 1e-4 --gpu_ids 2 --use_chop
#python train.py --opt options/train/BaselineCCA.yaml --name BaselineCCAx4_bs16ps96lr1e-4 --scale 4 --bs 16 --ps 96 --lr 1e-4 --gpu_ids 0 --use_chop
#python train.py --opt options/train/BaselineSA.yaml --name BaselineSAx4_bs16ps96lr1e-4 --scale 4 --bs 16 --ps 96 --lr 1e-4 --gpu_ids 1 --use_chop
#python train.py --opt options/train/BaselineESA.yaml --name BaselineESAx4_bs16ps96lr1e-4 --scale 4 --bs 16 --ps 96 --lr 1e-4 --gpu_ids 2 --use_chop
#python train.py --opt options/train/HFAN_D4U2.yaml --name HFAN_comparex4_bs16ps96lr1e-4 --scale 4 --bs 16 --ps 96 --lr 1e-4 --gpu_ids 3 --use_chop


#-------------------------------Study U---------------------------------
#python train.py --opt options/train/HFAN_D4U1.yaml --name HFAN_D4U1x4_bs16ps96lr6e-4 --scale 4 --bs 16 --ps 96 --lr 6e-4 --gpu_ids 2 --use_chop
#python train.py --opt options/train/HFAN_D4U2.yaml --name HFAN_D4U2x4_bs16ps96lr6e-4 --scale 4 --bs 16 --ps 96 --lr 6e-4 --gpu_ids 0 --use_chop
#python train.py --opt options/train/HFAN_D4U3.yaml --name HFAN_D4U3x4_bs16ps96lr6e-4 --scale 4 --bs 16 --ps 96 --lr 6e-4 --gpu_ids 1 --use_chop
#python train.py --opt options/train/HFAN_D4U4.yaml --name HFAN_D4U4x4_bs16ps96lr6e-4 --scale 4 --bs 16 --ps 96 --lr 6e-4 --gpu_ids 0 --use_chop
#python train.py --opt options/train/EDSR_ref.yaml --name EDSR_refx4 --scale 4 --bs 16 --ps 96 --lr 1e-4 --gpu_ids 1,2 --use_chop


#-------------------------------Study D---------------------------------
#python train.py --opt options/train/HFAN_D4U2.yaml --name HFAN_D4U2x4_bs16ps96lr6e-4 --scale 4 --bs 16 --ps 96 --lr 6e-4 --gpu_ids 0 --use_chop
#python train.py --opt options/train/HFAN_D6U2.yaml --name HFAN_D6U2x4_bs16ps96lr6e-4 --scale 4 --bs 16 --ps 96 --lr 6e-4 --gpu_ids 1 --use_chop
#python train.py --opt options/train/HFAN_D8U2.yaml --name HFAN_D8U2x4_bs16ps96lr6e-4 --scale 4 --bs 16 --ps 96 --lr 6e-4 --gpu_ids 2 --use_chop
#python train.py --opt options/train/HFAN_D10U2.yaml --name HFAN_D10U2x4_bs16ps96lr6e-4 --scale 4 --bs 16 --ps 96 --lr 6e-4 --gpu_ids 0 --use_chop
