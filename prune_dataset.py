

#%%
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.algorithms.hash_key_inference.prune import Prune
from datumaro.plugins.validators import DetectionValidator, SegmentationValidator
img_dir = "/home/lin/codebase/instance_segmentation/dataset_to_sample"
img_dir2 = "/home/lin/codebase/experiment_for_image_dataspliter/mixed_dataset"

env = Environment()
detected_format = env.detect_dataset(path=img_dir2)

# %%
dataset = Dataset.import_from(img_dir2, detected_format[0])

#%%

#%%  ###  cluster_random  ###
prune = Prune(dataset, cluster_method="cluster_random")
cluster_random_result = prune.get_pruned(0.5)

#%%


#%%
validator = SegmentationValidator()
cluster_random_reports = validator.validate(cluster_random_result)

cluster_random_stats = cluster_random_reports["statistics"]

label_stats = cluster_random_stats["label_distribution"]["defined_labels"]
label_name, label_counts = zip(*[(k, v) for k, v in label_stats.items()])


#%%
plt.figure(figsize=(12, 4))
plt.hist(label_name, weights=label_counts, bins=len(label_name))
plt.xticks(rotation="vertical")
plt.show()


#%%
repsave = "/home/lin/codebase/instance_segmentation/cocoa-ripeness-inst.v2i.coco-segmentation/repsample_cluster_random"
cluster_random_result.export(repsave, format="coco_instances", save_media=True)


#%%  ####  use query_clust method  ###
## query clust method needs to be debugged for valueerror
prune = Prune(dataset, cluster_method="query_clust")

#%%
quuery_clust_result = prune.get_pruned(0.5)

#%%  ####  centroid  ####
prune = Prune(dataset, cluster_method="centroid")
centroid_result = prune.get_pruned(0.5)

#%%
repsave = "/home/lin/codebase/instance_segmentation/repsample_centroid"
centroid_result.export(repsave, format="coco_instances", save_media=True)


#%%
"""
Epoch 19/19 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 270/270 0:02:05 • 0:00:00 2.18it/s v_num: 0 train/loss_rpn_cls: 0.000              
                                                                                        train/loss_rpn_bbox: 0.002 train/loss_cls: 0.015
                                                                                        train/loss_bbox: 0.020 train/loss_mask: 0.037   
                                                                                        train/loss: 0.074 validation/data_time: 0.005   
                                                                                        validation/iter_time: 0.083 val/map: 0.667      
                                                                                        val/map_50: 0.730 val/map_75: 0.720             
                                                                                        val/map_small: 0.201 val/map_medium: 0.534      
                                                                                        val/map_large: 0.711 val/mar_1: 0.652           
                                                                                        val/mar_10: 0.897 val/mar_100: 0.897            
                                                                                        val/mar_small: 0.322 val/mar_medium: 0.860      
                                                                                        val/mar_large: 0.953 val/map_per_class: -1.000  
                                                                                        val/mar_100_per_class: -1.000 val/f1-score:     
                                                                                        0.711 train/data_time: 0.012 train/iter_time:   
                                                                                        0.461                                           
Elapsed time: 0:56:42.451325

"""

## adjustment needs to be made for pruning to take and prune on train and val
# keep test separate as it is currently assummed that all the dataset is 
# being combined before pruning hence may not be comparable in terms of 
# getting a test dataset

#%%
centroid_reports = validator.validate(centroid_result)

centroid_stats = centroid_reports["statistics"]

label_stats = centroid_stats["label_distribution"]["defined_labels"]
label_name, label_counts = zip(*[(k, v) for k, v in label_stats.items()])

plt.figure(figsize=(12, 4))
plt.hist(label_name, weights=label_counts, bins=len(label_name))
plt.xticks(rotation="vertical")
plt.show()

#%%  ### Entropy  ###
prune = Prune(dataset, cluster_method="entropy")

#%%
entropy_result = prune.get_pruned(0.5)

#%%
entropy_reports = validator.validate(entropy_result)

entropy_stats = entropy_reports["statistics"]

label_stats = entropy_stats["label_distribution"]["defined_labels"]
label_name, label_counts = zip(*[(k, v) for k, v in label_stats.items()])

plt.figure(figsize=(12, 4))
plt.hist(label_name, weights=label_counts, bins=len(label_name))
plt.xticks(rotation="vertical")
plt.show()


# %%  ### Near duplicate removal
prune = Prune(dataset, cluster_method="ndr")
ndr_result = prune.get_pruned(0.5)

#%%
repsave = "/home/lin/codebase/instance_segmentation/repsample_ndr"

ndr_result.export(repsave, format="coco_instances", save_media=True)




# %%
ndr_reports = validator.validate(ndr_result)

ndr_stats = ndr_reports["statistics"]

label_stats = ndr_stats["label_distribution"]["defined_labels"]
label_name, label_counts = zip(*[(k, v) for k, v in label_stats.items()])

plt.figure(figsize=(12, 4))
plt.hist(label_name, weights=label_counts, bins=len(label_name))
plt.xticks(rotation="vertical")
plt.show()
# %%
random_result.export("random_result", format="datumaro", save_media=True)
cluster_random_result.export("cluster_random_result", format="datumaro", save_media=True)
#query_clust_result.export("query_clust_result", format="datumaro", save_media=True)
centroid_result.export("centroid_result", format="datumaro", save_media=True)
entropy_result.export("entropy_result", format="datumaro", save_media=True)
ndr_result.export("ndr_result", format="datumaro", save_media=True)

#%%
random_result.export("random_result_imagenetFormat", format="imagenet_with_subset_dirs", save_media=True)

# %%   ###train model with full data

# %%  experiment to determine if representative sampling reduces training time while 
# minimally compromising on the accuracy at an acceptable level

 
#%%

datapath = "/home/lin/codebase/instance_segmentation/cocoa-ripeness-inst.v2i.coco-segmentation"
dm.Dataset.from_import(datapath)

# %%
from glob import glob

val_dir = "/home/lin/codebase/instance_segmentation/cocoa-ripeness-inst.v2i.coco-segmentation/repsample_centroid/images/val"

len(glob(f"{val_dir}/*"))
# %%
"""
ALL DATA TRAINING RESULT

Epoch 16/19 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 603/603 0:04:21 • 0:00:00 2.23it/s v_num: 0 train/loss_rpn_cls: 0.002               
                                                                                        train/loss_rpn_bbox: 0.002 train/loss_cls: 0.011 
                                                                                        train/loss_bbox: 0.025 train/loss_mask: 0.040    
                                                                                        train/loss: 0.079 validation/data_time: 0.005    
                                                                                        validation/iter_time: 0.087 val/map: 0.720       
                                                                                        val/map_50: 0.778 val/map_75: 0.771              
                                                                                        val/map_small: 0.198 val/map_medium: 0.565       
                                                                                        val/map_large: 0.775 val/mar_1: 0.641 val/mar_10:
                                                                                        0.910 val/mar_100: 0.910 val/mar_small: 0.549    
                                                                                        val/mar_medium: 0.882 val/mar_large: 0.960       
                                                                                        val/map_per_class: -1.000 val/mar_100_per_class: 
                                                                                        -1.000 val/f1-score: 0.683 train/data_time: 0.013
                                                                                        train/iter_time: 0.433                           
Elapsed time: 1:40:05.112484

Testing results 

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test/data_time       │   0.0038684236351400614   │
│       test/f1-score       │    0.7150904536247253     │
│      test/iter_time       │   0.058442387729883194    │
│         test/map          │    0.7164003849029541     │
│        test/map_50        │    0.7748006582260132     │
│        test/map_75        │    0.7619205117225647     │
│      test/map_large       │    0.7924619317054749     │
│      test/map_medium      │    0.5712788701057434     │
│    test/map_per_class     │           -1.0            │
│      test/map_small       │    0.3468170464038849     │
│        test/mar_1         │    0.6534634232521057     │
│        test/mar_10        │    0.9099258780479431     │
│       test/mar_100        │     0.912447452545166     │
│  test/mar_100_per_class   │           -1.0            │
│      test/mar_large       │     0.961567223072052     │
│      test/mar_medium      │    0.8701627850532532     │
│      test/mar_small       │    0.6988954544067383     │
└───────────────────────────┴───────────────────────────┘
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 822/822 0:00:53 • 0:00:00 17.49it/s  
Elapsed time: 0:01:22.079637
"""


#%%


"""
# cluster random representative sampling results

Epoch 19/19 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 307/307 0:02:16 • 0:00:00 2.21it/s v_num: 0 train/loss_rpn_cls: 0.002 train/loss_rpn_bbox: 
                                                                                        0.002 train/loss_cls: 0.006 train/loss_bbox: 0.009      
                                                                                        train/loss_mask: 0.054 train/loss: 0.073                
                                                                                        validation/data_time: 0.005 validation/iter_time: 0.103 
                                                                                        val/map: 0.739 val/map_50: 0.804 val/map_75: 0.792      
                                                                                        val/map_small: 0.235 val/map_medium: 0.602              
                                                                                        val/map_large: 0.795 val/mar_1: 0.657 val/mar_10: 0.904 
                                                                                        val/mar_100: 0.905 val/mar_small: 0.617 val/mar_medium: 
                                                                                        0.881 val/mar_large: 0.951 val/map_per_class: -1.000    
                                                                                        val/mar_100_per_class: -1.000 val/f1-score: 0.729       
                                                                                        train/data_time: 0.013 train/iter_time: 0.442           
Elapsed time: 0:58:05.186021


TEST results of cluster random

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test/data_time       │   0.0037418848369270563   │
│       test/f1-score       │    0.7046895027160645     │
│      test/iter_time       │    0.05530881881713867    │
│         test/map          │     0.71424400806427      │
│        test/map_50        │    0.7701919078826904     │
│        test/map_75        │    0.7625433802604675     │
│      test/map_large       │    0.7809850573539734     │
│      test/map_medium      │    0.5688297152519226     │
│    test/map_per_class     │           -1.0            │
│      test/map_small       │    0.31389012932777405    │
│        test/mar_1         │     0.65065598487854      │
│        test/mar_10        │    0.9067968130111694     │
│       test/mar_100        │    0.9072619080543518     │
│  test/mar_100_per_class   │           -1.0            │
│      test/mar_large       │    0.9602509140968323     │
│      test/mar_medium      │    0.8677533864974976     │
│      test/mar_small       │    0.7368831634521484     │
└───────────────────────────┴───────────────────────────┘
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 822/822 0:00:46 • 0:00:00 18.43it/s  
Elapsed time: 0:00:59.623982
"""

#%%
"""
Centroid representative sampling results

Epoch 19/19 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 251/251 0:01:43 • 0:00:00 2.40it/s v_num: 0 train/loss_rpn_cls: 0.007 train/loss_rpn_bbox: 
                                                                                        0.007 train/loss_cls: 0.044 train/loss_bbox: 0.056      
                                                                                        train/loss_mask: 0.058 train/loss: 0.171                
                                                                                        validation/data_time: 0.005 validation/iter_time: 0.087 
                                                                                        val/map: 0.678 val/map_50: 0.735 val/map_75: 0.728      
                                                                                        val/map_small: 0.255 val/map_medium: 0.528              
                                                                                        val/map_large: 0.738 val/mar_1: 0.653 val/mar_10: 0.905 
                                                                                        val/mar_100: 0.906 val/mar_small: 0.586 val/mar_medium: 
                                                                                        0.862 val/mar_large: 0.955 val/map_per_class: -1.000    
                                                                                        val/mar_100_per_class: -1.000 val/f1-score: 0.677       
                                                                                        train/data_time: 0.013 train/iter_time: 0.409           
Elapsed time: 0:55:46.656891

TESTING centroid results

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test/data_time       │   0.0038126669824123383   │
│       test/f1-score       │    0.6628275513648987     │
│      test/iter_time       │    0.0571708083152771     │
│         test/map          │    0.6659988164901733     │
│        test/map_50        │     0.724271833896637     │
│        test/map_75        │    0.7149537205696106     │
│      test/map_large       │    0.7207879424095154     │
│      test/map_medium      │    0.5416668653488159     │
│    test/map_per_class     │           -1.0            │
│      test/map_small       │    0.29591119289398193    │
│        test/mar_1         │    0.6466283202171326     │
│        test/mar_10        │    0.9128502011299133     │
│       test/mar_100        │    0.9151757955551147     │
│  test/mar_100_per_class   │           -1.0            │
│      test/mar_large       │    0.9567306637763977     │
│      test/mar_medium      │    0.8778390288352966     │
│      test/mar_small       │    0.7745555639266968     │
└───────────────────────────┴───────────────────────────┘
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 822/822 0:00:47 • 0:00:00 17.94it/s  
Elapsed time: 0:01:01.012386
"""


#%%
"""
Near duplicate removal sampling (0.5) results

Epoch 19/19 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 302/302 0:02:07 • 0:00:00 2.38it/s v_num: 0 train/loss_rpn_cls: 0.000           
                                                                                        train/loss_rpn_bbox: 0.001 train/loss_cls:   
                                                                                        0.009 train/loss_bbox: 0.027 train/loss_mask:
                                                                                        0.046 train/loss: 0.083 validation/data_time:
                                                                                        0.005 validation/iter_time: 0.094 val/map:   
                                                                                        0.701 val/map_50: 0.758 val/map_75: 0.750    
                                                                                        val/map_small: 0.203 val/map_medium: 0.614   
                                                                                        val/map_large: 0.768 val/mar_1: 0.648        
                                                                                        val/mar_10: 0.897 val/mar_100: 0.899         
                                                                                        val/mar_small: 0.364 val/mar_medium: 0.873   
                                                                                        val/mar_large: 0.954 val/map_per_class:      
                                                                                        -1.000 val/mar_100_per_class: -1.000         
                                                                                        val/f1-score: 0.688 train/data_time: 0.015   
                                                                                        train/iter_time: 0.421                       
Elapsed time: 0:58:51.582670

Testing NDR result
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test/data_time       │   0.003897094866260886    │
│       test/f1-score       │    0.6940667629241943     │
│      test/iter_time       │    0.05886773020029068    │
│         test/map          │    0.6986624002456665     │
│        test/map_50        │    0.7592872381210327     │
│        test/map_75        │    0.7470318675041199     │
│      test/map_large       │    0.7724733948707581     │
│      test/map_medium      │    0.5428561568260193     │
│    test/map_per_class     │           -1.0            │
│      test/map_small       │    0.34717413783073425    │
│        test/mar_1         │     0.64734947681427      │
│        test/mar_10        │    0.9140156507492065     │
│       test/mar_100        │     0.915643572807312     │
│  test/mar_100_per_class   │           -1.0            │
│      test/mar_large       │    0.9584324359893799     │
│      test/mar_medium      │    0.8819739818572998     │
│      test/mar_small       │    0.7879408001899719     │
└───────────────────────────┴───────────────────────────┘
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 822/822 0:00:50 • 0:00:00 17.22it/s  
Elapsed time: 0:01:04.011446
"""


#%%