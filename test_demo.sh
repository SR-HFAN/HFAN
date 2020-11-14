#------------------------------EDSR_CVPRW2017---------------------------
#python test.py --opt options/test/EDSR.yaml --name EDSRx2 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 2 --gpu_ids 0 --which_model EDSR --pretrained experiment/EDSRx2_bs16ps96lr1e-4/epochs/best.pth
#python test.py --opt options/test/EDSR.yaml --name EDSRx3 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 3 --gpu_ids 0 --which_model EDSR --pretrained experiment/EDSRx3_bs16ps96lr1e-4/epochs/best.pth
#python test.py --opt options/test/EDSR.yaml --name EDSRx4 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 4 --gpu_ids 0 --which_model EDSR --pretrained experiment/EDSRx4_bs16ps96lr1e-4/epochs/best.pth

#----------------------------IMDN_ACM_MM2019----------------------------
#python test.py --opt options/test/IMDN.yaml --name IMDNx2 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 2 --gpu_ids 0 --which_model IMDN --pretrained experiment/IMDNx2_bs16ps96lr2e-4/epochs/best.pth
#python test.py --opt options/test/IMDN.yaml --name IMDNx3 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 3 --gpu_ids 0 --which_model IMDN --pretrained experiment/IMDNx3_bs16ps96lr2e-4/epochs/best.pth
#python test.py --opt options/test/IMDN.yaml --name IMDNx4 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 4 --gpu_ids 0 --which_model IMDN --pretrained experiment/IMDNx4_bs16ps96lr2e-4/epochs/best.pth

#---------------------------LatticeNet_ECCV2020-------------------------
#python test.py --opt options/test/LatticeNet.yaml --name LatticeNetx2 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 2 --gpu_ids 0 --which_model LatticeNet --pretrained experiment/LatticeNetx2_bs16ps96F64lr2e-4/epochs/best.pth
#python test.py --opt options/test/LatticeNet.yaml --name LatticeNetx3 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 3 --gpu_ids 0 --which_model LatticeNet --pretrained experiment/LatticeNetx3_bs16ps96lr2e-4/epochs/best.pth
#python test.py --opt options/test/LatticeNet.yaml --name LatticeNetx4 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 4 --gpu_ids 0 --which_model LatticeNet --pretrained experiment/LatticeNetx4_bs16ps96lr2e-4/epochs/best.pth

#-----------------------------CARN_ECCV2018-----------------------------
#python test.py --opt options/test/CARN.yaml --name CARNx2 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 2 --gpu_ids 0 --which_model CARN --pretrained experiment/CARNx2_bs16ps96lr1e-4/epochs/best.pth
#python test.py --opt options/test/CARN.yaml --name CARNx3 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 3 --gpu_ids 0 --which_model CARN --pretrained experiment/CARNx3_bs16ps96lr1e-4/epochs/best.pth
#python test.py --opt options/test/CARN.yaml --name CARNx4 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 4 --gpu_ids 0 --which_model CARN --pretrained experiment/CARNx4_bs16ps96lr1e-4/epochs/best.pth

#-----------------------------IDN_CVPR2018------------------------------
#python test.py --opt options/test/IDN.yaml --name IDNx2 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 2 --gpu_ids 0 --which_model IDN --pretrained experiment/IDNx2_bs16ps96lr2e-4/epochs/best.pth
#python test.py --opt options/test/IDN.yaml --name IDNx3 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 3 --gpu_ids 0 --which_model IDN --pretrained experiment/IDNx3_bs16ps96lr2e-4/epochs/best.pth
#python test.py --opt options/test/IDN.yaml --name IDNx4 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 4 --gpu_ids 0 --which_model IDN --pretrained experiment/IDNx4_bs16ps96lr2e-4/epochs/best.pth

#---------------------------------HFAN----------------------------------
#python test.py --opt options/test/HFAN.yaml --name HFANx2 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 2 --gpu_ids 0 --which_model HFAN --pretrained experiment/HFANx2_bs16ps96lr6e-4/epochs/best.pth
#python test.py --opt options/test/HFAN.yaml --name HFANx3 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 3 --gpu_ids 0 --which_model HFAN --pretrained experiment/HFANx3_bs16ps96lr6e-4/epochs/best.pth
#python test.py --opt options/test/HFAN.yaml --name HFANx4 --dataset_name Set5+Set14+B100+Urban100+Manga109 --scale 4 --gpu_ids 0 --which_model HFAN --pretrained experiment/HFANx4_bs16ps96lr6e-4/epochs/best.pth

#-----------Compare with embedded attention mechanism-------------------
#python test.py --opt options/test/Baseline.yaml --name Baselinex4 --dataset_name Set5 --scale 4 --gpu_ids 0 --which_model Baseline --pretrained experiment/Baselinex4_bs16ps96lr1e-4/epochs/best.pth
#python test.py --opt options/test/BaselineCA.yaml --name BaselineCAx4 --dataset_name Set5 --scale 4 --gpu_ids 0 --which_model BaselineCA --pretrained experiment/BaselineCAx4_bs16ps96lr1e-4/epochs/best.pth
#python test.py --opt options/test/BaselineCCA.yaml --name BaselineCCAx4 --dataset_name Set5 --scale 4 --gpu_ids 0 --which_model BaselineCCA --pretrained experiment/BaselineCCAx4_bs16ps96lr1e-4/epochs/best.pth
#python test.py --opt options/test/BaselineSA.yaml --name BaselineSAx4 --dataset_name Set5 --scale 4 --gpu_ids 0 --which_model BaselineSA --pretrained experiment/BaselineSAx4_bs16ps96lr1e-4/epochs/best.pth
#python test.py --opt options/test/BaselineESA.yaml --name BaselineESAx4 --dataset_name Set5 --scale 4 --gpu_ids 0 --which_model BaselineESA --pretrained experiment/BaselineESAx4_bs16ps96lr1e-4/epochs/best.pth
#python test.py --opt options/test/HFAN.yaml --name HFAN_comparex4 --dataset_name Set5 --scale 4 --gpu_ids 0 --which_model HFAN --pretrained experiment/HFAN_comparex4_bs16ps96lr1e-4/epochs/best.pth
