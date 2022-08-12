# this file include all kinds of util function for analysis script

def from_image_id_to_file_name(image_id):
    # this function convert the image_id in the coco dataset 
    # to the respective file name
    result = str(image_id).zfill(12)
    return result

    