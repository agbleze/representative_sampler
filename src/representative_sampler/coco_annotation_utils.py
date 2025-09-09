import os
from glob import glob
import shutil
import json
from copy import deepcopy
from pandas import json_normalize

def combine_data(img_dir_list, coco_ann_filepath_list, 
                 save_img_dir_as, save_combined_coco_ann_as
                 ):
    all_imgs_path = []
    for imgdir in img_dir_list:
        imgs = [imgfile_path for imgfile_path in glob(f"{imgdir}/*") 
                if imgfile_path.split(".")[-1] != "json"
                ]
        all_imgs_path.extend(imgs)
    os.makedirs(save_img_dir_as, exist_ok=True)
    for img in all_imgs_path:
        shutil.copy(img, save_img_dir_as)
        
    combined_ann = combine_annotations(list_of_annotation_paths=coco_ann_filepath_list,
                                        save_annotation_as=save_combined_coco_ann_as,
                                        info_description="Coco annotations"
                                        )
    return combined_ann
    
    
def combine_annotations(list_of_annotation_paths, save_annotation_as, info_description):
    tag_categories = list()
    merged_coco = {
                    "info": {"description": info_description},
                    "licenses": [],
                    "images": [],
                    "annotations": [],
                    "categories": [],
                    "tag_categories": tag_categories
                }
    print("started combine annotations")
    image_id_records = []
    annotation_id_record = []
    categories = get_unique_categories(list_of_annotation_paths)
    catname_catid_map = {cat_obj["name"]: cat_obj["id"] for cat_obj in categories}
    merged_coco["categories"] = categories
    print(f"list_of_annotation_paths: {list_of_annotation_paths}")
    print(f"save_annotation_as: {save_annotation_as}")
    for file_path in list_of_annotation_paths:
        with open(file_path, "r") as f:
            coco_data = json.load(f)
        current_file_categories = coco_data["categories"]
        current_catid_catname_map = {cat_obj["id"]: cat_obj["name"] for cat_obj in current_file_categories}
        
        try:
            tag_categories += coco_data["tag_categories"]
            print(f"tag_categories present in file hence used")
        except KeyError:
            print(f"tag_categories not present in file")
        for image in coco_data["images"]:
            image_id = image["id"]
            filename = image["file_name"]
            
            if image_id not in image_id_records:
                merged_coco["images"].append(image)
                image_id_records.append(image_id)
                
                annot_for_img_id = [annot for annot in coco_data["annotations"]
                                    if image_id==annot["image_id"]
                                    ]
                for ann in annot_for_img_id:
                    if ann["id"] not in annotation_id_record:
                        ann_catname = current_catid_catname_map[ann["category_id"]]
                        ann["category_id"] = catname_catid_map[ann_catname]
                        merged_coco["annotations"].append(ann)    
                        annotation_id_record.append(ann["id"])
                    else:
                        ann_increase = max(annotation_id_record) +1 
                        ann["id"] = ann_increase
                        ann_catname = current_catid_catname_map[ann["category_id"]]
                        ann["category_id"] = catname_catid_map[ann_catname]
                        
                        merged_coco["annotations"].append(ann)
                        annotation_id_record.append(ann_increase)
            else:
                annot_for_img_id = [annot for annot in deepcopy(coco_data["annotations"])
                                    if int(image_id)==int(annot["image_id"])
                                    ]
                image_id_inc = max(image_id_records) + 1
                image["id"] = image_id_inc
                merged_coco["images"].append(image)
                image_id_records.append(image_id_inc)
                
                update_list = []
                for ann in annot_for_img_id:
                    if ann["image_id"] not in image_id_records:
                        print(f"{ann['image_id']} not in {image_id_records}")
                    else:
                        ann["image_id"] = image_id_inc
                        
                    if ann["id"] not in annotation_id_record:
                        ann_catname = current_catid_catname_map[ann["category_id"]]
                        ann["category_id"] = catname_catid_map[ann_catname]
                    else:
                        new_annot = max(annotation_id_record) + 1
                        ann["id"] = new_annot
                        ann_catname = current_catid_catname_map[ann["category_id"]]
                        ann["category_id"] = catname_catid_map[ann_catname]
                        annotation_id_record.append(new_annot)
                    update_list.append(ann)
                merged_coco["annotations"].extend(update_list)
    with open(save_annotation_as, "w") as f:
        json.dump(merged_coco, f)
    print(f"Completed combine annotation")
    
    corrected_cocodata = scrutinize_and_correct_annotation(coco_annotation_file=save_annotation_as,
                                                            save_annotation_as=save_annotation_as
                                                            )
    print("Completed scrutinize and correct annotation")
    return corrected_cocodata
    
    
def get_unique_categories(json_files):
    categories = []
    for file_index, file_path in enumerate(json_files):
        if file_index == 0:
            with open(file_path, "r") as f:
                file_annotation = json.load(f)
            unique_available_classname = []
            for category in file_annotation["categories"]:
                if category["name"] not in unique_available_classname:
                    unique_available_classname.append(category["name"])
                    categories.append(category)
        else:
            with open(file_path, 'r') as f:
                file_annotation = json.load(f)
                file_categories = file_annotation['categories']
            for cat in file_categories:
                if cat["name"] in unique_available_classname:
                    print(f"{cat['name']} already exists in {json_files[0]} hence not used to update categories")
                else:
                    categories.append(cat)
                    unique_available_classname.append(cat["name"])
    return categories
            
            
def tidy_imgnames_in_annotation(coco_annotation_file, save_annotation_as):
    with open(coco_annotation_file, "r") as f:
        coco_data = json.load(f)
    _coco_data = deepcopy(coco_data)
    images = []
    for img in _coco_data["images"]:
        img["file_name"] = os.path.basename(img["file_name"])
        images.append(img)
    tidied_coco_data = {}
    for field in _coco_data:
        if field != "images":
            tidied_coco_data[field] = _coco_data[field]
    tidied_coco_data["images"] = images
    with open(save_annotation_as, "w") as f:
        json.dump(tidied_coco_data, f)
    return tidied_coco_data

def tidy_duplicate_category(coco_annotation_file, save_annotation_as):
    with open(coco_annotation_file, "r") as f:
        coco_data = json.load(f)
    _coco_data = deepcopy(coco_data)
    actual_catid_catname_map = {}
    for category in _coco_data["categories"]:
        actual_catid_catname_map[category["id"]] = category["name"]
        
    category_names = {}
    categories = []
    for category in _coco_data["categories"]:
        if category["name"] not in category_names.keys():
            catid_assign = len(category_names.keys()) + 1
            category_names[category["name"]] = catid_assign
            category["id"] = catid_assign
            categories.append(category)
            
    annotations = []
    for ann in _coco_data["annotations"]:
        ann_category_id = ann["category_id"]
        actual_catname = actual_catid_catname_map[ann_category_id]
        catid_assign = category_names[actual_catname]
        ann["category_id"] = catid_assign
        annotations.append(ann)
        
    coco_data_without_duplicates = {}
    for field in _coco_data.keys():
        if field not in ["annotations", "categories"]:
            coco_data_without_duplicates[field] = _coco_data[field]
    coco_data_without_duplicates["categories"] = categories
    coco_data_without_duplicates["annotations"] = annotations
    with open(save_annotation_as, "w") as f:
        json.dump(coco_data_without_duplicates, f)
    return coco_data_without_duplicates

def make_annotation_ids_unique(coco_annotation_file, save_annotation_as):
    with open(coco_annotation_file, "r") as f:
        coco_data = json.load(f)
    annotation_id_record = []
    annotations = []
    _coco_data = deepcopy(coco_data)
    for ann in coco_data["annotations"]:
        if ann["id"] not in annotation_id_record:
            annotations.append(ann)
            annotation_id_record.append(ann["id"])
        else:
            annotation_id_assign = max(annotation_id_record) + 1
            ann["id"] = annotation_id_assign
            annotations.append(ann)
            annotation_id_record.append(ann["id"])
            
    unique_annotation_id_coco = {}
    for field in _coco_data:
        if field != "annotations":
            unique_annotation_id_coco[field] = _coco_data[field]
    unique_annotation_id_coco["annotations"] = annotations
    with open(save_annotation_as, "w") as f:
        json.dump(unique_annotation_id_coco, f)
    return unique_annotation_id_coco
    
def make_images_ids_unique(coco_annotation_file, save_annotation_as):
    with open(coco_annotation_file, "r") as f:
        coco_data = json.load(f)
    images_id_record = []
    images = []
    annotations = []
    _coco_data = deepcopy(coco_data)
    for img in coco_data["images"]:
        if img["id"] not in images_id_record:
            images.append(img)
            images_id_record.append(img["id"])
            
            for ann in coco_data["annotations"]:
                if img["id"] == ann["image_id"]:
                    annotations.append(ann)
        else:
            image_id_assign = max(images_id_record) + 1
            current_img_id = img["id"]
            img["id"] = image_id_assign
            images.append(img)
            
            for ann in coco_data["annotations"]:
                if ann["image_id"] == current_img_id:
                    ann["image_id"] = image_id_assign
                    annotations.append(ann)
            images_id_record.append(image_id_assign)
    unique_image_id_coco = {}
    for field in _coco_data:
        if field not in ["annotations", "images"]:
            unique_image_id_coco[field] = _coco_data[field]
    unique_image_id_coco["annotations"] = annotations
    unique_image_id_coco["images"] = images
    with open(save_annotation_as, "w") as f:
        json.dump(unique_image_id_coco, f)
    return unique_image_id_coco


def scrutinize_and_correct_annotation(coco_annotation_file,
                                      save_annotation_as
                                    ):
    tidy_imgname_cocodata = tidy_imgnames_in_annotation(coco_annotation_file,
                                                        save_annotation_as
                                                        )
    tidy_duplicate_catcocodata = tidy_duplicate_category(save_annotation_as,
                                                         save_annotation_as
                                                         )
    unique_annid_cocodata = make_annotation_ids_unique(save_annotation_as,
                                                       save_annotation_as
                                                       )
    unique_imgid_cocodata = make_images_ids_unique(save_annotation_as,
                                                   save_annotation_as
                                                   )
    return unique_imgid_cocodata

def subset_coco_annotations(img_list, coco_annotation_file,
                            save_annotations_as
                            ):
    with open(coco_annotation_file, "r") as f:
        coco_data = json.load(f)
        
    name_to_id = {os.path.basename(image["file_name"]): image["id"] for image in coco_data["images"]}
    image_ids = [name_to_id[os.path.basename(name)] for name in img_list if os.path.basename(name) in name_to_id]
    
    annotations = []
    for annotation in coco_data["annotations"]:
        if annotation["image_id"] in image_ids:
            annotations.append(annotation)
    try:
        tag_categories = coco_data["tag_categories"]
    except KeyError:
        tag_categories = [{}]
        
    try:
        info = coco_data["info"]
    except KeyError:
        info = {}
        
    try:
        licenses = coco_data["licenses"]
    except:
        licenses = {}
    images = []
    for image in coco_data["images"]:
        if image["id"] in image_ids:
            images.append(image)
            
    subset_json = {"info": info,
                   "licenses": licenses,
                   "categories": coco_data["categories"],
                   "images": images,
                   "annotations": annotations,
                   "tag_categories": tag_categories
                   }
    with open(save_annotations_as, "w") as f:
        json.dump(subset_json, f)
    return subset_json


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
