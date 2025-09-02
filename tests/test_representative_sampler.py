from representative_sampler import representative_sampler
import pytest
import os
import tempfile
from cpauger.generate_coco_ann import generate_random_images_and_annotation



tempdir = tempfile.TemporaryDirectory()
train_img_dir = os.path.join(tempdir.name,"train_random_images")
train_coco_json_file=os.path.join(tempdir.name,"train_generated_annotation.json")
test_img_dir=os.path.join(tempdir.name,"test_random_images")
test_coco_json_file=os.path.join(tempdir.name, "test_generated_annotation.json")
train_data_name="random_train"
test_data_name="random_test"
output_dir=os.path.join(tempdir.name, "random_model_train")
save_class_metadata_as = os.path.join(tempdir.name, "class_metadata_map.json")


train_imgpaths, train_coco_path = generate_random_images_and_annotation(image_height=224, image_width=224,
                                                                        number_of_images=10, 
                                                                        output_dir=train_img_dir,
                                                                        img_ext ="jpg",
                                                                        image_name="train_random_images",
                                                                        parallelize=True,
                                                                        save_ann_as=train_coco_json_file,
                                                                        )


test_imgpaths, test_coco_path = generate_random_images_and_annotation(image_height=224, image_width=224,
                                                                    number_of_images=5, 
                                                                    output_dir=test_img_dir,
                                                                    img_ext ="jpg",
                                                                    image_name="test_random_images",
                                                                    parallelize=True,
                                                                    save_ann_as=test_coco_json_file,
                                                                    )

def test_sample_data():
    pass
