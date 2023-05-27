# this script aims to calculate the statistic number of the CLIP
# the average iou, overall overlap area, overlap ratio

import json
import torch
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
iou_calculator = BboxOverlaps2D()

# load the gt bboxes
#gt_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_train2017_except_48base_only.json'))
gt_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_train2017_0_8000.json'))
#gt_content = json.load(open('/data/zhuoming/detection/lvis_v1/annotations/lvis_v1_train_0_8000.json'))
#gt_content = json.load(open('/project/nevatia_174/zhuoming/detection/lvis_v1/annotations/lvis_v1_train_0_8000.json'))

#gt_content = json.load(open('/project/nevatia_174/zhuoming/detection/lvis_v1/annotations/lvis_v1_train_8000_16000.json'))

base_names = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush')

novel_names = ('airplane', 'bus', 'cat', 'dog', 'cow', 
            'elephant', 'umbrella', 'tie', 'snowboard', 
            'skateboard', 'cup', 'knife', 'cake', 'couch', 
            'keyboard', 'sink', 'scissors')

#base_names = ['air_conditioner', 'airplane', 'alarm_clock', 'antenna', 'apple', 'apron', 'armchair', 'trash_can', 'avocado', 'awning', 'baby_buggy', 'backpack', 'handbag', 'suitcase', 'ball', 'balloon', 'banana', 'bandanna', 'banner', 'barrel', 'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap', 'baseball_glove', 'basket', 'bath_mat', 'bath_towel', 'bathtub', 'beanie', 'bear', 'bed', 'bedspread', 'cow', 'beef_(food)', 'beer_bottle', 'bell', 'bell_pepper', 'belt', 'belt_buckle', 'bench', 'bicycle', 'visor', 'billboard', 'bird', 'birthday_cake', 'blackboard', 'blanket', 'blender', 'blinker', 'blouse', 'blueberry', 'boat', 'bolt', 'book', 'boot', 'bottle', 'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'box', 'bracelet', 'bread', 'bridal_gown', 'broccoli', 'bucket', 'bun', 'buoy', 'bus_(vehicle)', 'butter', 'button', 'cab_(taxi)', 'cabinet', 'cake', 'calendar', 'camera', 'can', 'candle', 'candle_holder', 'cap_(headwear)', 'bottle_cap', 'car_(automobile)', 'railcar_(part_of_a_train)', 'carrot', 'tote_bag', 'cat', 'cauliflower', 'celery', 'cellular_telephone', 'chair', 'chandelier', 'choker', 'chopping_board', 'chopstick', 'Christmas_tree', 'cigarette', 'cistern', 'clock', 'clock_tower', 'coaster', 'coat', 'coffee_maker', 'coffee_table', 'computer_keyboard', 'condiment', 'cone', 'control', 'cookie', 'cooler_(for_food)', 'cork_(bottle_plug)', 'edible_corn', 'cowboy_hat', 'crate', 'crossbar', 'crumb', 'cucumber', 'cup', 'cupboard', 'cupcake', 'curtain', 'cushion', 'deck_chair', 'desk', 'dining_table', 'dish', 'dishtowel', 'dishwasher', 'dispenser', 'Dixie_cup', 'dog', 'dog_collar', 'doll', 'doorknob', 'doughnut', 'drawer', 'dress', 'dress_suit', 'dresser', 'duck', 'duffel_bag', 'earphone', 'earring', 'egg', 'refrigerator', 'elephant', 'fan', 'faucet', 'figurine', 'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fireplace', 'fireplug', 'fish', 'flag', 'flagpole', 'flip-flop_(sandal)', 'flower_arrangement', 'fork', 'frisbee', 'frying_pan', 'giraffe', 'glass_(drink_container)', 'glove', 'goggles', 'grape', 'green_bean', 'green_onion', 'grill', 'guitar', 'hairbrush', 'ham', 'hair_dryer', 'hand_towel', 'handle', 'hat', 'headband', 'headboard', 'headlight', 'helmet', 'hinge', 'home_plate_(baseball)', 'fume_hood', 'hook', 'horse', 'hose', 'polar_bear', 'iPod', 'jacket', 'jar', 'jean', 'jersey', 'key', 'kitchen_sink', 'kite', 'knee_pad', 'knife', 'knob', 'ladder', 'lamb_(animal)', 'lamp', 'lamppost', 'lampshade', 'lanyard', 'laptop_computer', 'latch', 'lemon', 'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb', 'lime', 'log', 'speaker_(stero_equipment)', 'magazine', 'magnet', 'mailbox_(at_home)', 'manhole', 'map', 'marker', 'mask', 'mast', 'mattress', 'microphone', 'microwave_oven', 'milk', 'minivan', 'mirror', 'monitor_(computer_equipment) computer_monitor', 'motor', 'motor_scooter', 'motorcycle', 'mound_(baseball)', 'mouse_(computer_equipment)', 'mousepad', 'mug', 'mushroom', 'napkin', 'necklace', 'necktie', 'newspaper', 'notebook', 'nut', 'oar', 'onion', 'orange_(fruit)', 'ottoman', 'oven', 'paddle', 'painting', 'pajamas', 'pan_(for_cooking)', 'paper_plate', 'paper_towel', 'parking_meter', 'pastry', 'pear', 'pen', 'pencil', 'pepper', 'person', 'piano', 'pickle', 'pickup_truck', 'pillow', 'pineapple', 'pipe', 'pitcher_(vessel_for_liquid)', 'pizza', 'place_mat', 'plate', 'pole', 'polo_shirt', 'pop_(soda)', 'poster', 'pot', 'flowerpot', 'potato', 'printer', 'propeller', 'quilt', 'radiator', 'rearview_mirror', 'reflector', 'remote_control', 'ring', 'rubber_band', 'plastic_bag', 'saddle_(on_an_animal)', 'saddle_blanket', 'sail', 'salad', 'saltshaker', 'sandal_(type_of_shoe)', 'sandwich', 'saucer', 'sausage', 'scale_(measuring_instrument)', 'scarf', 'scissors', 'scoreboard', 'scrubbing_brush', 'sheep', 'shirt', 'shoe', 'shopping_bag', 'short_pants', 'shoulder_bag', 'shower_head', 'shower_curtain', 'signboard', 'sink', 'skateboard', 'ski', 'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'snowboard', 'soap', 'soccer_ball', 'sock', 'sofa', 'soup', 'spatula', 'spectacles', 'spoon', 'statue_(sculpture)', 'steering_wheel', 'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign', 'streetlight', 'suit_(clothing)', 'sunglasses', 'surfboard', 'sweater', 'sweatshirt', 'swimsuit', 'table', 'tablecloth', 'tag', 'taillight', 'tank_(storage_vessel)', 'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tarp', 'teapot', 'teddy_bear', 'telephone', 'telephone_pole', 'television_set', 'tennis_ball', 'tennis_racket', 'thermostat', 'tinfoil', 'tissue_paper', 'toaster', 'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toothbrush', 'toothpaste', 'toothpick', 'cover', 'towel', 'towel_rack', 'toy', 'traffic_light', 'trailer_truck', 'train_(railroad_vehicle)', 'tray', 'tripod', 'trousers', 'truck', 'umbrella', 'underwear', 'urinal', 'vase', 'vent', 'vest', 'wall_socket', 'wallet', 'watch', 'water_bottle', 'watermelon', 'weathervane', 'wet_suit', 'wheel', 'windshield_wiper', 'wine_bottle', 'wineglass', 'blinder_(for_horses)', 'wristband', 'wristlet', 'zebra']
#novel_names = ['aerosol_can', 'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'aquarium', 'armband', 'artichoke', 'ashtray', 'asparagus', 'atomizer', 'award', 'basketball_backboard', 'bagel', 'bamboo', 'Band_Aid', 'bandage', 'barrette', 'barrow', 'basketball', 'bat_(animal)', 'bathrobe', 'battery', 'bead', 'bean_curd', 'beanbag', 'beer_can', 'beret', 'bib', 'binder', 'binoculars', 'birdfeeder', 'birdbath', 'birdcage', 'birdhouse', 'black_sheep', 'blackberry', 'blazer', 'bobbin', 'bobby_pin', 'boiled_egg', 'deadbolt', 'bookcase', 'booklet', 'bottle_opener', 'bouquet', 'bowler_hat', 'suspenders', 'brassiere', 'bread-bin', 'briefcase', 'broom', 'brownie', 'brussels_sprouts', 'bull', 'bulldog', 'bullet_train', 'bulletin_board', 'bullhorn', 'bunk_bed', 'business_card', 'butterfly', 'cabin_car', 'calculator', 'calf', 'camcorder', 'camel', 'camera_lens', 'camper_(vehicle)', 'can_opener', 'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup', 'cape', 'cappuccino', 'identity_card', 'card', 'cardigan', 'horse_carriage', 'cart', 'carton', 'cash_register', 'cast', 'cayenne_(spice)', 'CD_player', 'cherry', 'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'crisp_(potato_chip)', 'chocolate_bar', 'chocolate_cake', 'slide', 'cigarette_case', 'clasp', 'cleansing_agent', 'clip', 'clipboard', 'clothes_hamper', 'clothespin', 'coat_hanger', 'coatrack', 'cock', 'coconut', 'coffeepot', 'coin', 'colander', 'coleslaw', 'pacifier', 'corkscrew', 'cornet', 'cornice', 'corset', 'costume', 'cowbell', 'crab_(animal)', 'cracker', 'crayon', 'crescent_roll', 'crib', 'crock_pot', 'crow', 'crown', 'crucifix', 'cruise_ship', 'police_cruiser', 'crutch', 'cub_(animal)', 'cube', 'cufflink', 'trophy_cup', 'dartboard', 'deer', 'dental_floss', 'diaper', 'dish_antenna', 'dishrag', 'dolphin', 'domestic_ass', 'doormat', 'underdrawers', 'dress_hat', 'drill', 'drum_(musical_instrument)', 'duckling', 'duct_tape', 'dumpster', 'eagle', 'easel', 'egg_yolk', 'eggbeater', 'eggplant', 'elk', 'envelope', 'eraser', 'Ferris_wheel', 'ferry', 'fighter_jet', 'file_cabinet', 'fire_hose', 'fish_(food)', 'fishing_rod', 'flamingo', 'flannel', 'flap', 'flashlight', 'flipper_(footwear)', 'flute_glass', 'foal', 'folding_chair', 'food_processor', 'football_(American)', 'footstool', 'forklift', 'freight_car', 'French_toast', 'freshener', 'frog', 'fruit_juice', 'garbage_truck', 'garden_hose', 'gargle', 'garlic', 'gazelle', 'gelatin', 'giant_panda', 'gift_wrap', 'ginger', 'cincture', 'globe', 'goat', 'golf_club', 'golfcart', 'goose', 'grater', 'gravestone', 'grizzly', 'grocery_bag', 'gull', 'gun', 'hairnet', 'hairpin', 'hamburger', 'hammer', 'hammock', 'hamster', 'handcart', 'handkerchief', 'veil', 'headscarf', 'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'highchair', 'hog', 'honey', 'hot_sauce', 'hummingbird', 'icecream', 'ice_maker', 'igniter', 'iron_(for_clothing)', 'ironing_board', 'jam', 'jeep', 'jet_plane', 'jewelry', 'jumpsuit', 'kayak', 'kettle', 'kilt', 'kimono', 'kitten', 'kiwi_fruit', 'ladle', 'ladybug', 'lantern', 'legging_(clothing)', 'Lego', 'lion', 'lip_balm', 'lizard', 'lollipop', 'loveseat', 'mail_slot', 'mandarin_orange', 'manger', 'mashed_potato', 'mat_(gym_equipment)', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine', 'melon', 'mitten', 'mixer_(kitchen_tool)', 'money', 'monkey', 'muffin', 'musical_instrument', 'needle', 'nest', 'newsstand', 'nightshirt', 'noseband_(for_animals)', 'notepad', 'oil_lamp', 'olive_oil', 'orange_juice', 'ostrich', 'overalls_(clothing)', 'owl', 'packet', 'pad', 'padlock', 'paintbrush', 'palette', 'pancake', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol', 'parka', 'parrot', 'passenger_car_(part_of_a_train)', 'passport', 'pea_(food)', 'peach', 'peanut_butter', 'peeler_(tool_for_fruit_and_vegetables)', 'pelican', 'penguin', 'pepper_mill', 'perfume', 'pet', 'pew_(church_bench)', 'phonograph_record', 'pie', 'pigeon', 'pinecone', 'pita_(bread)', 'platter', 'pliers', 'pocketknife', 'poker_(fire_stirring_tool)', 'pony', 'postbox_(public)', 'postcard', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel', 'projectile_(weapon)', 'projector', 'pumpkin', 'puppy', 'rabbit', 'racket', 'radio_receiver', 'radish', 'raft', 'raincoat', 'ram_(animal)', 'raspberry', 'razorblade', 'reamer_(juicer)', 'receipt', 'recliner', 'record_player', 'rhinoceros', 'rifle', 'robe', 'rocking_chair', 'rolling_pin', 'router_(computer_equipment)', 'runner_(carpet)', 'saddlebag', 'salami', 'salmon_(fish)', 'salsa', 'school_bus', 'screwdriver', 'sculpture', 'seabird', 'seahorse', 'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark', 'shaving_cream', 'shield', 'shopping_cart', 'shovel', 'silo', 'skewer', 'sled', 'sleeping_bag', 'slipper_(footwear)', 'snowman', 'snowmobile', 'solar_array', 'soupspoon', 'sour_cream', 'spice_rack', 'spider', 'sponge', 'sportswear', 'spotlight', 'squirrel', 'stapler_(stapling_machine)', 'starfish', 'steak_(food)', 'step_stool', 'stereo_(sound_system)', 'strainer', 'sunflower', 'sunhat', 'sushi', 'mop', 'sweat_pants', 'sweatband', 'sweet_potato', 'sword', 'table_lamp', 'tape_measure', 'tapestry', 'tartan', 'tassel', 'tea_bag', 'teacup', 'teakettle', 'telephone_booth', 'television_camera', 'thermometer', 'thermos_bottle', 'thread', 'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinsel', 'toast_(food)', 'toolbox', 'tortilla', 'tow_truck', 'tractor_(farm_equipment)', 'dirt_bike', 'tricycle', 'trunk', 'turban', 'turkey_(food)', 'turtle', 'turtleneck_(clothing)', 'typewriter', 'urn', 'vacuum_cleaner', 'vending_machine', 'videotape', 'volleyball', 'waffle', 'wagon', 'wagon_wheel', 'walking_stick', 'wall_clock', 'automatic_washer', 'water_cooler', 'water_faucet', 'water_jug', 'water_scooter', 'water_ski', 'water_tower', 'watering_can', 'webcam', 'wedding_cake', 'wedding_ring', 'wheelchair', 'whipped_cream', 'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)', 'windsock', 'wine_bucket', 'wok', 'wooden_spoon', 'wreath', 'wrench', 'yacht', 'yogurt', 'yoke_(animal_equipment)', 'zucchini']

novel_id = []
base_id = []
for ele in gt_content['categories']:
    category_id = ele['id']
    name = ele['name']
    if name in base_names:
        base_id.append(category_id)
    elif name in novel_names:
        novel_id.append(category_id)

# aggreate the anotation base on the image id
from_image_id_to_annotation = {}
for anno in gt_content['annotations']:
    image_id = anno['image_id']
    cate_id = anno['category_id']
    bbox = anno['bbox']
    bbox.append(cate_id)
    if image_id not in from_image_id_to_annotation:
        from_image_id_to_annotation[image_id] = {'base':[], 'novel':[]}
    if cate_id in base_id:
        from_image_id_to_annotation[image_id]['base'].append(bbox)
    #elif cate_id in novel_id:
    #    from_image_id_to_annotation[image_id]['novel'].append(bbox)
    else:
        from_image_id_to_annotation[image_id]['novel'].append(bbox)

# collect the image info
from_image_id_to_image_info = {}
for info in gt_content['images']:
    image_id = info['id']
    from_image_id_to_image_info[image_id] = info

# load the proposal and print the image
#save_root = '/home/zhuoming/coco_visualization_most_matched'
proposal_path_root = ['/data/zhuoming/detection/coco/clip_proposal/32_32_512', 
                      '/data/zhuoming/detection/coco/rpn_proposal/mask_rcnn_r50_fpn_2x_coco_2gpu_base48_reg_class_agno']


for root in proposal_path_root:
    all_iou_for_novel = 0
    all_overlap_on_novel = 0
    all_valid_dist_area_novel = 0
    image_with_novel = 0
    over_03_novel = 0
    over_05_novel = 0
    over_08_novel = 0
    
    all_iou_for_base = 0
    all_overlap_on_base = 0
    all_valid_dist_area_base = 0
    image_with_base = 0
    over_03_base = 0
    over_05_base = 0
    over_08_base = 0
    
    for i, image_id in enumerate(from_image_id_to_annotation):
        base_gt_bboxes = torch.tensor(from_image_id_to_annotation[image_id]['base'])
        novel_gt_bboxes = torch.tensor(from_image_id_to_annotation[image_id]['novel'])
        if base_gt_bboxes.shape[0] == 0 and novel_gt_bboxes.shape[0] == 0:
            continue
        #download the image
        file_name = from_image_id_to_image_info[image_id]['coco_url'].split('/')[-1]

        pregenerate_prop_path = os.path.join(root, '.'.join(file_name.split('.')[:-1]) + '.json')
        #try:
        if not os.path.exists(pregenerate_prop_path):
            continue
            
        pregenerated_bbox = json.load(open(pregenerate_prop_path))
        #except:
        #    continue
        # read the proposal directly from the proposal file
        all_proposals = torch.tensor(pregenerated_bbox['score'])
        
        # for extract proposal from the feature file 
        # all_proposals = torch.tensor(pregenerated_bbox['bbox'])
        # ## scale the bbox back to the original size
        # img_metas = pregenerated_bbox['img_metas']
        # pre_extract_scale_factor = np.array(img_metas['scale_factor']+[1,1]).astype(np.float32)
        # # all_proposals: [562.7451782226562, 133.49032592773438, 653.2548217773438, 314.5096740722656, 0.9763965010643005, 461.0]
        # all_proposals = all_proposals / pre_extract_scale_factor
        
        # find the matched bbox for gt bbox "novel"
        if novel_gt_bboxes.shape[0] != 0:
            # convert the gt for the novel
            novel_gt_bboxes[:, 2] = novel_gt_bboxes[:, 0] + novel_gt_bboxes[:, 2] 
            novel_gt_bboxes[:, 3] = novel_gt_bboxes[:, 1] + novel_gt_bboxes[:, 3] 
            proposal_with_novel_iou = iou_calculator(all_proposals[:, :4], novel_gt_bboxes)
            accumulate_proposal_with_novel_iou = torch.mean(proposal_with_novel_iou[proposal_with_novel_iou != 0])
            if torch.isnan(accumulate_proposal_with_novel_iou):
                #print(proposal_with_novel_iou[proposal_with_novel_iou != 0])
                accumulate_proposal_with_novel_iou = 0
            all_iou_for_novel += accumulate_proposal_with_novel_iou
            image_with_novel += 1
            
            # calculate the overall overlap with the gt bboxes
            overlap_on_novel_gt = iou_calculator(novel_gt_bboxes, all_proposals[:, :4], 'iof')
            accumulate_overlap_on_novel_gt = torch.mean(overlap_on_novel_gt[overlap_on_novel_gt != 0])
            if torch.isnan(accumulate_overlap_on_novel_gt):
                #print(overlap_on_novel_gt[overlap_on_novel_gt != 0])
                accumulate_overlap_on_novel_gt = 0
            all_overlap_on_novel += accumulate_overlap_on_novel_gt
            the_max_per_proposal = torch.max(overlap_on_novel_gt, dim=0)
            over_08_novel += torch.sum(the_max_per_proposal>0.8)
            over_05_novel += torch.sum(the_max_per_proposal>0.5)
            over_03_novel += torch.sum(the_max_per_proposal>0.3)
            
            # calculate the overall overlap with the proposal
            valid_distill_area = iou_calculator(all_proposals[:, :4], novel_gt_bboxes, 'iof')
            accumulate_valid_distill_area = torch.mean(valid_distill_area[valid_distill_area != 0])
            if torch.isnan(accumulate_valid_distill_area):
                #print(overlap_on_novel_gt[overlap_on_novel_gt != 0])
                accumulate_valid_distill_area = 0
            all_valid_dist_area_novel += accumulate_valid_distill_area
        
        if base_gt_bboxes.shape[0] != 0:
            # convert the gt for the base
            base_gt_bboxes[:, 2] = base_gt_bboxes[:, 0] + base_gt_bboxes[:, 2] 
            base_gt_bboxes[:, 3] = base_gt_bboxes[:, 1] + base_gt_bboxes[:, 3] 
            proposal_with_base_iou = iou_calculator(all_proposals[:, :4], base_gt_bboxes)
            accumulate_proposal_with_base_iou = torch.mean(proposal_with_base_iou[proposal_with_base_iou != 0])
            all_iou_for_base += accumulate_proposal_with_base_iou
            image_with_base += 1 
            
            # calculate the overall overlap with the gt bboxes
            overlap_on_base_gt = iou_calculator(base_gt_bboxes, all_proposals[:, :4], 'iof')
            accumulate_overlap_on_base_gt = torch.mean(overlap_on_base_gt[overlap_on_base_gt != 0])
            all_overlap_on_base += accumulate_overlap_on_base_gt
            the_max_per_proposal = torch.max(overlap_on_base_gt, dim=0)
            over_08_base += torch.sum(the_max_per_proposal>0.8)
            over_05_base += torch.sum(the_max_per_proposal>0.5)
            over_03_base += torch.sum(the_max_per_proposal>0.3)

            # calculate the overall overlap with the proposal
            valid_distill_area = iou_calculator(all_proposals[:, :4], base_gt_bboxes, 'iof')
            accumulate_valid_distill_area = torch.mean(valid_distill_area[valid_distill_area != 0])
            if torch.isnan(accumulate_valid_distill_area):
                #print(overlap_on_novel_gt[overlap_on_novel_gt != 0])
                accumulate_valid_distill_area = 0
            all_valid_dist_area_base += accumulate_valid_distill_area          
    print('image_with_novel', image_with_novel, 'image_with_base', image_with_base)
    print(root, 'avg iou for novel:', all_iou_for_novel / image_with_novel, 'avg iou for base:', all_iou_for_base / image_with_base,
          'avg overlap on gt for novel:', all_overlap_on_novel / image_with_novel, 'avg overlap on gt for base:', all_overlap_on_base / image_with_base,
          'all_valid_dist_area_novel:', all_valid_dist_area_novel / image_with_novel, 'all_valid_dist_area_base:', all_valid_dist_area_base / image_with_base,
          'over_08_novel:', over_08_novel, 'over_05_novel:', over_05_novel, 'over_03_novel', over_03_novel, 
          'over_08_base', over_08_base, 'over_05_base', over_05_base, 'over_03_base', over_03_base)
