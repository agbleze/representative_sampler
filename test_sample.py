#%%
from representative_sampler.representative_sampler import (sample_data, 
                                                           RepresentativenessSampler, 
                                                           ImageUniqueness,
                                                           Entropy, ClusteredRandom,
                                                           Centroid, 
                                                           )
from representative_sampler.coco_annotation_utils import subset_coco_annotations
import os
import json
from glob import glob
import shutil

# %%
train_img_dir = "/home/lin/codebase/instance_segmentation/dataset_to_sample/images/train"
val_img_dir = "/home/lin/codebase/instance_segmentation/dataset_to_sample/images/val"

train_coco_annfile = "/home/lin/codebase/instance_segmentation/dataset_to_sample/annotations/instances_train.json"
val_coco_annfile = "/home/lin/codebase/instance_segmentation/dataset_to_sample/annotations/instances_val.json"

train_img_list = glob(f"{train_img_dir}/*")
val_img_list = glob(f"{val_img_dir}/*")

# %%
rep_meanshift_local = RepresentativenessSampler(img_list=train_img_list,
                                                cluster_algorithm="meanshift",
                                                bandwidth=0.9
                                                )
# %%
meanshift_local_samples = rep_meanshift_local.sample(sample_ratio=0.5)
# %%
meanshift_img_train_dir = "/home/lin/codebase/instance_segmentation/meanshift_repsentative_local/images/train"
# %%
sampled_train_meanshift_local = [obj[0] for obj in meanshift_local_samples]
# %%

meanshif_annotations = "/home/lin/codebase/instance_segmentation/meanshift_repsentative_local/annotations/instances_train.json"
subset_coco_annotations(img_list=sampled_train_meanshift_local,
                        coco_annotation_file=train_coco_annfile,
                        save_annotations_as=meanshif_annotations
                        )



for img in sampled_train_meanshift_local:
    shutil.copy(img, meanshift_img_train_dir)


# %%   ########  val ###### 

val_rep_meanshift_local = RepresentativenessSampler(img_list=val_img_list,
                                                cluster_algorithm="meanshift",
                                                bandwidth=0.9
                                                )

val_meanshift_local_samples = val_rep_meanshift_local.sample(sample_ratio=0.5)

meanshift_img_val_dir = "/home/lin/codebase/instance_segmentation/meanshift_repsentative_local/images/val"

sampled_val_meanshift_local = [obj[0] for obj in val_meanshift_local_samples]

import shutil
val_meanshift_annotations = "/home/lin/codebase/instance_segmentation/meanshift_repsentative_local/annotations/instances_val.json"
subset_coco_annotations(img_list=sampled_val_meanshift_local,
                        coco_annotation_file=val_coco_annfile,
                        save_annotations_as=val_meanshift_annotations
                        )



for img in sampled_val_meanshift_local:
    shutil.copy(img, meanshift_img_val_dir)
    
    
#%%
from typing import Literal

def sample_rep_data(img_list,
                    destination_img_dir,
                    cluster_algorithm="meanshift",
                    bandwidth=0.9,
                    sample_ratio=0.5,
                    coco_annotation_file=None,
                    save_annotations_as=None,
                    scoring_method: Literal['cluster-center', 'cluster-center-downweight'] = "cluster-center",
                    kmeans_n_clusters: int = 20,
                    bandwidth_quantile: float = 0.8,
                    bandwidth_n_samples: int = 500,
                    norm_method: Literal['local', 'global'] = "local"
                    ):
    os.makedirs(destination_img_dir, exist_ok=True)
    save_ann_dir = os.path.dirname(save_annotations_as)
    os.makedirs(save_ann_dir, exist_ok=True)
    repsampler = RepresentativenessSampler(img_list=img_list,
                                            cluster_algorithm=cluster_algorithm,
                                            bandwidth=bandwidth,
                                            norm_method=norm_method,
                                            n_samples=bandwidth_n_samples,
                                            quantile=bandwidth_quantile,
                                            n_clusters=kmeans_n_clusters,
                                            method=scoring_method
                                            )
    samples_obj = repsampler.sample(sample_ratio=sample_ratio)

    selected_imgs = [obj[0] for obj in 
                    samples_obj
                    ]
    if coco_annotation_file:
        subset_coco_annotations(img_list=selected_imgs,
                                coco_annotation_file=coco_annotation_file,
                                save_annotations_as=save_annotations_as
                                )

    for img in selected_imgs:
        shutil.copy(img, destination_img_dir)
        
# %%  ##############  sample images for meanshift global norm_method
sample_rep_data(img_list=train_img_list,
                cluster_algorithm="meanshift",
                coco_annotation_file=train_coco_annfile,
                save_annotations_as="/home/lin/codebase/instance_segmentation/meanshift_repsentative_global/annotations/instances_train.json",
                destination_img_dir="/home/lin/codebase/instance_segmentation/meanshift_repsentative_global/images/train",
                norm_method="global"
                )

#%%
sample_rep_data(img_list=val_img_list,
                cluster_algorithm="meanshift",
                coco_annotation_file=val_coco_annfile,
                save_annotations_as="/home/lin/codebase/instance_segmentation/meanshift_repsentative_global/annotations/instances_val.json",
                destination_img_dir="/home/lin/codebase/instance_segmentation/meanshift_repsentative_global/images/val",
                norm_method="global"
                )
#%%


# TODO:
# REPEAT FOR global 
# use kmeans with n_clusters = num_classes and num of samples to reduce to

# %%  ##############  sample images for kmeans n_clusters = sample_size, norm_method=global
#####  scoring_method = 'cluster-center'

int(len(train_img_list)*0.5)
#%%
sample_rep_data(img_list=train_img_list,
                cluster_algorithm="kmeans",
                coco_annotation_file=train_coco_annfile,
                save_annotations_as="/home/lin/codebase/instance_segmentation/kmeans_repsentative_global_ncluster_samsize/annotations/instances_train.json",
                destination_img_dir="/home/lin/codebase/instance_segmentation/kmeans_repsentative_global_ncluster_samsize/images/train",
                norm_method="global",
                kmeans_n_clusters=int(len(train_img_list)*0.5)
                )


sample_rep_data(img_list=val_img_list,
                cluster_algorithm="kmeans",
                coco_annotation_file=val_coco_annfile,
                save_annotations_as="/home/lin/codebase/instance_segmentation/kmeans_repsentative_global_ncluster_samsize/annotations/instances_val.json",
                destination_img_dir="/home/lin/codebase/instance_segmentation/kmeans_repsentative_global_ncluster_samsize/images/val",
                norm_method="global",
                kmeans_n_clusters=int(len(val_img_list)*0.5)
                )

#%%

# %%  ##############  sample images for kmeans n_clusters = sample_size,   ##############
# norm_method=local scoring_method = 'cluster-center'

#%%
sample_ratio = 0.5
sample_rep_data(img_list=train_img_list,
                cluster_algorithm="kmeans",
                coco_annotation_file=train_coco_annfile,
                save_annotations_as="/home/lin/codebase/instance_segmentation/kmeans_repsentative_local_ncluster_samsize/annotations/instances_train.json",
                destination_img_dir="/home/lin/codebase/instance_segmentation/kmeans_repsentative_local_ncluster_samsize/images/train",
                norm_method="local",
                kmeans_n_clusters=int(len(train_img_list)*sample_ratio)
                )


sample_rep_data(img_list=val_img_list,
                cluster_algorithm="kmeans",
                coco_annotation_file=val_coco_annfile,
                save_annotations_as="/home/lin/codebase/instance_segmentation/kmeans_repsentative_local_ncluster_samsize/annotations/instances_val.json",
                destination_img_dir="/home/lin/codebase/instance_segmentation/kmeans_repsentative_local_ncluster_samsize/images/val",
                norm_method="local",
                kmeans_n_clusters=int(len(val_img_list)*sample_ratio)
                )


#%%
from pandas import json_normalize

def coco_annotation_to_df(coco_annotation_file):
    with open(coco_annotation_file, "r") as annot_file:
        annotation = json.load(annot_file)
    annotations_df = json_normalize(annotation, "annotations")
    annot_imgs_df = json_normalize(annotation, "images")
    annot_cat_df = json_normalize(annotation, "categories")
    annotations_images_merge_df = annotations_df.merge(annot_imgs_df, left_on='image_id', 
                                                        right_on='id',
                                                        suffixes=("_annotation", "_image"),
                                                        how="outer"
                                                        )
    annotations_imgs_cat_merge = annotations_images_merge_df.merge(annot_cat_df, left_on="category_id", right_on="id",
                                                                    suffixes=(None, '_categories'),
                                                                    how="outer"
                                                                    )
    all_merged_df = annotations_imgs_cat_merge[['id_annotation', 'image_id','category_id', 'bbox', 'area', 'segmentation', 'iscrowd',
                                'file_name', 'height', 'width', 'name', 'supercategory'
                                ]]
    all_merged_df.rename(columns={"name": "category_name",
                                  "height": "image_height",
                                  "width": "image_width"}, 
                         inplace=True
                         )
    all_merged_df.dropna(subset=["file_name"], inplace=True)
    return all_merged_df

#%%

train_df = coco_annotation_to_df(train_coco_annfile)

val_df = coco_annotation_to_df(val_coco_annfile)
#%%

train_num_categories = train_df.category_id.nunique()
val_num_categories = val_df.category_id.nunique()

#%%  ###

# %%  ###  sample images for kmeans n_clusters = num_categories,   ##############
# norm_method=local scoring_method = 'cluster-center'


sample_rep_data(img_list=train_img_list,
                cluster_algorithm="kmeans",
                coco_annotation_file=train_coco_annfile,
                save_annotations_as="/home/lin/codebase/instance_segmentation/kmeans_repsentative_local_ncluster_numcat/annotations/instances_train.json",
                destination_img_dir="/home/lin/codebase/instance_segmentation/kmeans_repsentative_local_ncluster_numcat/images/train",
                norm_method="local",
                kmeans_n_clusters=train_num_categories
                )


sample_rep_data(img_list=val_img_list,
                cluster_algorithm="kmeans",
                coco_annotation_file=val_coco_annfile,
                save_annotations_as="/home/lin/codebase/instance_segmentation/kmeans_repsentative_local_ncluster_numcat/annotations/instances_val.json",
                destination_img_dir="/home/lin/codebase/instance_segmentation/kmeans_repsentative_local_ncluster_numcat/images/val",
                norm_method="local",
                kmeans_n_clusters=val_num_categories
                )


# %%  ###  sample images for kmeans n_clusters = num_categories,   ##############
# norm_method=global scoring_method = 'cluster-center'


sample_rep_data(img_list=train_img_list,
                cluster_algorithm="kmeans",
                coco_annotation_file=train_coco_annfile,
                save_annotations_as="/home/lin/codebase/instance_segmentation/kmeans_repsentative_global_ncluster_numcat/annotations/instances_train.json",
                destination_img_dir="/home/lin/codebase/instance_segmentation/kmeans_repsentative_global_ncluster_numcat/images/train",
                norm_method="global",
                kmeans_n_clusters=train_num_categories
                )

sample_rep_data(img_list=val_img_list,
                cluster_algorithm="kmeans",
                coco_annotation_file=val_coco_annfile,
                save_annotations_as="/home/lin/codebase/instance_segmentation/kmeans_repsentative_global_ncluster_numcat/annotations/instances_val.json",
                destination_img_dir="/home/lin/codebase/instance_segmentation/kmeans_repsentative_global_ncluster_numcat/images/val",
                norm_method="global",
                kmeans_n_clusters=val_num_categories
                )

#%%  ######    Image Uniqueness  #########

img_uniq = ImageUniqueness(img_list=train_img_list,
                           nearest_neighbour_influence=3,
                           neighbour_weights=[0.6,0.3,0.1]
                           )

#%%

sample_img_uniq_obj = img_uniq.sample(sample_ratio=0.5)

#%%

uniq_id_sample_imgs = [obj[0] for obj in sample_img_uniq_obj]


#%%

def export_data(coco_annotation_file, destination_img_dir,
                save_annotations_as, img_list
                ):
    os.makedirs(destination_img_dir, exist_ok=True)
    save_ann_dir = os.path.dirname(save_annotations_as)
    os.makedirs(save_ann_dir, exist_ok=True)
    if coco_annotation_file:
        subset_coco_annotations(img_list=img_list,
                                coco_annotation_file=coco_annotation_file,
                                save_annotations_as=save_annotations_as
                                )

    for img in img_list:
        shutil.copy(img, destination_img_dir)


#%%
export_data(img_list=uniq_id_sample_imgs,
            coco_annotation_file=train_coco_annfile,
            save_annotations_as="/home/lin/codebase/instance_segmentation/uniqueness_repsentative/annotations/instances_train.json",
            destination_img_dir="/home/lin/codebase/instance_segmentation/uniqueness_repsentative/images/train"
            )
        


#%%  uniqueness val  ###
val_img_uniq = ImageUniqueness(img_list=val_img_list,
                                nearest_neighbour_influence=3,
                                neighbour_weights=[0.6,0.3,0.1]
                                )

val_sample_img_uniq_obj = val_img_uniq.sample(sample_ratio=0.5)

val_uniq_id_sample_imgs = [obj[0] for obj in val_sample_img_uniq_obj]

export_data(img_list=val_uniq_id_sample_imgs,
            coco_annotation_file=val_coco_annfile,
            save_annotations_as="/home/lin/codebase/instance_segmentation/uniqueness_repsentative/annotations/instances_val.json",
            destination_img_dir="/home/lin/codebase/instance_segmentation/uniqueness_repsentative/images/val"
            )
        

#%%
train_entr = Entropy(img_list=train_img_list,
                    n_clusters=int(len(train_img_list)*0.5)
                    #train_num_categories
                    #int(len(train_img_list)*0.5),
                    )

#%%
train_entr_selected_imgs = train_entr.base(alpha=0.7)

#%%

train_entr


#%%
from sklearn.cluster import KMeans
from representative_sampler.representative_sampler import EmbeddingExtractor

#%%

emtr = EmbeddingExtractor(img_list=train_img_list, model_type="ViT-B/32")


normalized_embedding = emtr._extract_img_features()
#%%
kmeans = KMeans(n_clusters=5, #num_centers, 
                random_state=0
                )
clusters = kmeans.fit_predict(normalized_embedding) #fit_predict(database_keys)


#%%

import numpy as np
cluster_ids, cluster_num_item_list = np.unique(clusters, return_counts=True)


#%%

for id, count in zip(cluster_ids, cluster_num_item_list):
    print(id, count)


#%%

cluster_ids[9]
#%%


sample_rep_data(img_list=train_img_list,
                cluster_algorithm="kmeans",
                coco_annotation_file=train_coco_annfile,
                save_annotations_as="/home/lin/codebase/instance_segmentation/kmeans_repsentative_global_ncluster_samsize/annotations/instances_train.json",
                destination_img_dir="/home/lin/codebase/instance_segmentation/kmeans_repsentative_global_ncluster_samsize/images/train",
                norm_method="global",
                kmeans_n_clusters=int(len(train_img_list)*0.5)
                )
        
# %%
os.path.dirname("/home/lin/codebase/instance_segmentation/kmeans_repsentative_global_ncluster_samsize/annotations/instances_val.json")
# %%
