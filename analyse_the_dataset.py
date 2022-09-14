# this script aims to analyse the dataset statistic
from __future__ import annotations
import json

# load the annotation
dataset_path = '/data/zhuoming/detection/coco/annotations/instances_train2017.json'
#dataset_path = '/data/zhuoming/detection/coco/annotations/instances_val2017.json'
#dataset_path = '/data/zhuoming/detection/lvis_v1/annotations/lvis_v1_train.json'
#dataset_path = '/data/zhuoming/detection/lvis_v1/annotations/lvis_v1_val.json'
dataset_content = json.load(open(dataset_path))

# present the number of total image 
print('number of image:', len(dataset_content['images']))

# define the base and novel categories name
base_cate_name = ('person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'microwave', 'oven', 'toaster', 'refrigerator', 'book', 'clock', 'vase', 'toothbrush')
novel_cate_name = ('airplane', 'bus', 'cat', 'dog', 'cow', 'elephant', 'umbrella', 'tie', 'snowboard', 'skateboard', 'cup', 'knife', 'cake', 'couch', 'keyboard', 'sink', 'scissors')

# base_cate_name = ['aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna', 'apple', 'apron', 'aquarium', 'armband', 'armchair', 'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer', 'avocado', 'award', 'awning', 'baby_buggy', 'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel', 'ball', 'balloon', 'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banner', 'barrel', 'barrette', 'barrow', 'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap', 'baseball_glove', 'basket', 'basketball', 'bat_(animal)', 'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'battery', 'bead', 'bean_curd', 'beanbag', 'beanie', 'bear', 'bed', 'bedspread', 'cow', 'beef_(food)', 'beer_bottle', 'beer_can', 'bell', 'bell_pepper', 'belt', 'belt_buckle', 'bench', 'beret', 'bib', 'bicycle', 'visor', 'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath', 'birdcage', 'birdhouse', 'birthday_cake', 'black_sheep', 'blackberry', 'blackboard', 'blanket', 'blazer', 'blender', 'blinker', 'blouse', 'blueberry', 'boat', 'bobbin', 'bobby_pin', 'boiled_egg', 'deadbolt', 'bolt', 'book', 'bookcase', 'booklet', 'boot', 'bottle', 'bottle_opener', 'bouquet', 'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'bowler_hat', 'box', 'suspenders', 'bracelet', 'brassiere', 'bread-bin', 'bread', 'bridal_gown', 'briefcase', 'broccoli', 'broom', 'brownie', 'brussels_sprouts', 'bucket', 'bull', 'bulldog', 'bullet_train', 'bulletin_board', 'bullhorn', 'bun', 'bunk_bed', 'buoy', 'bus_(vehicle)', 'business_card', 'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabin_car', 'cabinet', 'cake', 'calculator', 'calendar', 'calf', 'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)', 'can', 'can_opener', 'candle', 'candle_holder', 'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino', 'car_(automobile)', 'railcar_(part_of_a_train)', 'identity_card', 'card', 'cardigan', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton', 'cash_register', 'cast', 'cat', 'cauliflower', 'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone', 'chair', 'chandelier', 'cherry', 'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'crisp_(potato_chip)', 'chocolate_bar', 'chocolate_cake', 'choker', 'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cigarette', 'cigarette_case', 'cistern', 'clasp', 'cleansing_agent', 'clip', 'clipboard', 'clock', 'clock_tower', 'clothes_hamper', 'clothespin', 'coaster', 'coat', 'coat_hanger', 'coatrack', 'cock', 'coconut', 'coffee_maker', 'coffee_table', 'coffeepot', 'coin', 'colander', 'coleslaw', 'pacifier', 'computer_keyboard', 'condiment', 'cone', 'control', 'cookie', 'cooler_(for_food)', 'cork_(bottle_plug)', 'corkscrew', 'edible_corn', 'cornet', 'cornice', 'corset', 'costume', 'cowbell', 'cowboy_hat', 'crab_(animal)', 'cracker', 'crate', 'crayon', 'crescent_roll', 'crib', 'crock_pot', 'crossbar', 'crow', 'crown', 'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch', 'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup', 'cupboard', 'cupcake', 'curtain', 'cushion', 'dartboard', 'deck_chair', 'deer', 'dental_floss', 'desk', 'diaper', 'dining_table', 'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher', 'dispenser', 'Dixie_cup', 'dog', 'dog_collar', 'doll', 'dolphin', 'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit', 'dresser', 'drill', 'drum_(musical_instrument)', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumpster', 'eagle', 'earphone', 'earring', 'easel', 'egg', 'egg_yolk', 'eggbeater', 'eggplant', 'refrigerator', 'elephant', 'elk', 'envelope', 'eraser', 'fan', 'faucet', 'Ferris_wheel', 'ferry', 'fighter_jet', 'figurine', 'file_cabinet', 'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace', 'fireplug', 'fish', 'fish_(food)', 'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap', 'flashlight', 'flip-flop_(sandal)', 'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal', 'folding_chair', 'food_processor', 'football_(American)', 'footstool', 'fork', 'forklift', 'freight_car', 'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice', 'frying_pan', 'garbage_truck', 'garden_hose', 'gargle', 'garlic', 'gazelle', 'gelatin', 'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture', 'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles', 'golf_club', 'golfcart', 'goose', 'grape', 'grater', 'gravestone', 'green_bean', 'green_onion', 'grill', 'grizzly', 'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet', 'hairpin', 'ham', 'hamburger', 'hammer', 'hammock', 'hamster', 'hair_dryer', 'hand_towel', 'handcart', 'handkerchief', 'handle', 'hat', 'veil', 'headband', 'headboard', 'headlight', 'headscarf', 'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet', 'highchair', 'hinge', 'hog', 'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'horse', 'hose', 'hot_sauce', 'hummingbird', 'polar_bear', 'icecream', 'ice_maker', 'igniter', 'iPod', 'iron_(for_clothing)', 'ironing_board', 'jacket', 'jam', 'jar', 'jean', 'jeep', 'jersey', 'jet_plane', 'jewelry', 'jumpsuit', 'kayak', 'kettle', 'key', 'kilt', 'kimono', 'kitchen_sink', 'kite', 'kitten', 'kiwi_fruit', 'knee_pad', 'knife', 'knob', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard', 'laptop_computer', 'latch', 'legging_(clothing)', 'Lego', 'lemon', 'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb', 'lime', 'lion', 'lip_balm', 'lizard', 'log', 'lollipop', 'speaker_(stero_equipment)', 'loveseat', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)', 'mandarin_orange', 'manger', 'manhole', 'map', 'marker', 'mashed_potato', 'mask', 'mast', 'mat_(gym_equipment)', 'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine', 'melon', 'microphone', 'microwave_oven', 'milk', 'minivan', 'mirror', 'mitten', 'mixer_(kitchen_tool)', 'money', 'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor', 'motor_scooter', 'motorcycle', 'mound_(baseball)', 'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom', 'musical_instrument', 'napkin', 'necklace', 'necktie', 'needle', 'nest', 'newspaper', 'newsstand', 'nightshirt', 'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'oar', 'oil_lamp', 'olive_oil', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich', 'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas', 'palette', 'pan_(for_cooking)', 'pancake', 'paper_plate', 'paper_towel', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol', 'parka', 'parking_meter', 'parrot', 'passenger_car_(part_of_a_train)', 'passport', 'pastry', 'pea_(food)', 'peach', 'peanut_butter', 'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'pelican', 'pen', 'pencil', 'penguin', 'pepper', 'pepper_mill', 'perfume', 'person', 'pet', 'pew_(church_bench)', 'phonograph_record', 'piano', 'pickle', 'pickup_truck', 'pie', 'pigeon', 'pillow', 'pineapple', 'pinecone', 'pipe', 'pita_(bread)', 'pitcher_(vessel_for_liquid)', 'pizza', 'place_mat', 'plate', 'platter', 'pliers', 'pocketknife', 'poker_(fire_stirring_tool)', 'pole', 'polo_shirt', 'pony', 'pop_(soda)', 'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel', 'printer', 'projectile_(weapon)', 'projector', 'propeller', 'pumpkin', 'puppy', 'quilt', 'rabbit', 'racket', 'radiator', 'radio_receiver', 'radish', 'raft', 'raincoat', 'ram_(animal)', 'raspberry', 'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt', 'recliner', 'record_player', 'reflector', 'remote_control', 'rhinoceros', 'rifle', 'ring', 'robe', 'rocking_chair', 'rolling_pin', 'router_(computer_equipment)', 'rubber_band', 'runner_(carpet)', 'plastic_bag', 'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'sail', 'salad', 'salami', 'salmon_(fish)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)', 'sandwich', 'saucer', 'sausage', 'scale_(measuring_instrument)', 'scarf', 'school_bus', 'scissors', 'scoreboard', 'screwdriver', 'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark', 'shaving_cream', 'sheep', 'shield', 'shirt', 'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shoulder_bag', 'shovel', 'shower_head', 'shower_curtain', 'signboard', 'silo', 'sink', 'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'sled', 'sleeping_bag', 'slipper_(footwear)', 'snowboard', 'snowman', 'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'solar_array', 'soup', 'soupspoon', 'sour_cream', 'spatula', 'spectacles', 'spice_rack', 'spider', 'sponge', 'spoon', 'sportswear', 'spotlight', 'squirrel', 'stapler_(stapling_machine)', 'starfish', 'statue_(sculpture)', 'steak_(food)', 'steering_wheel', 'step_stool', 'stereo_(sound_system)', 'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer', 'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign', 'streetlight', 'suit_(clothing)', 'sunflower', 'sunglasses', 'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband', 'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword', 'table', 'table_lamp', 'tablecloth', 'tag', 'taillight', 'tank_(storage_vessel)', 'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure', 'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup', 'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth', 'telephone_pole', 'television_camera', 'television_set', 'tennis_ball', 'tennis_racket', 'thermometer', 'thermos_bottle', 'thermostat', 'thread', 'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil', 'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush', 'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel', 'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light', 'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'tray', 'tricycle', 'tripod', 'trousers', 'truck', 'trunk', 'turban', 'turkey_(food)', 'turtle', 'turtleneck_(clothing)', 'typewriter', 'umbrella', 'underwear', 'urinal', 'urn', 'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest', 'videotape', 'volleyball', 'waffle', 'wagon', 'wagon_wheel', 'walking_stick', 'wall_clock', 'wall_socket', 'wallet', 'automatic_washer', 'watch', 'water_bottle', 'water_cooler', 'water_faucet', 'water_jug', 'water_scooter', 'water_ski', 'water_tower', 'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake', 'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream', 'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)', 'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket', 'wineglass', 'blinder_(for_horses)', 'wok', 'wooden_spoon', 'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt', 'yoke_(animal_equipment)', 'zebra', 'zucchini']
# novel_cate_name = ['applesauce', 'apricot', 'arctic_(type_of_shoe)', 'armoire', 'armor', 'ax', 'baboon', 'bagpipe', 'baguet', 'bait', 'ballet_skirt', 'banjo', 'barbell', 'barge', 'bass_horn', 'batter_(food)', 'beachball', 'bedpan', 'beeper', 'beetle', 'Bible', 'birthday_card', 'pirate_flag', 'blimp', 'gameboard', 'bob', 'bolo_tie', 'bonnet', 'bookmark', 'boom_microphone', 'bow_(weapon)', 'pipe_bowl', 'bowling_ball', 'boxing_glove', 'brass_plaque', 'breechcloth', 'broach', 'bubble_gum', 'horse_buggy', 'bulldozer', 'bulletproof_vest', 'burrito', 'cabana', 'locker', 'candy_bar', 'canteen', 'elevator_car', 'car_battery', 'cargo_ship', 'carnation', 'casserole', 'cassette', 'chain_mail', 'chaise_longue', 'chalice', 'chap', 'checkbook', 'checkerboard', 'chessboard', 'chime', 'chinaware', 'poker_chip', 'chocolate_milk', 'chocolate_mousse', 'cider', 'cigar_box', 'clarinet', 'cleat_(for_securing_rope)', 'clementine', 'clippers_(for_plants)', 'cloak', 'clutch_bag', 'cockroach', 'cocoa_(beverage)', 'coil', 'coloring_material', 'combination_lock', 'comic_book', 'compass', 'convertible_(automobile)', 'sofa_bed', 'cooker', 'cooking_utensil', 'corkboard', 'cornbread', 'cornmeal', 'cougar', 'coverall', 'crabmeat', 'crape', 'cream_pitcher', 'crouton', 'crowbar', 'hair_curler', 'curling_iron', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'date_(fruit)', 'detergent', 'diary', 'die', 'dinghy', 'tux', 'dishwasher_detergent', 'diving_board', 'dollar', 'dollhouse', 'dove', 'dragonfly', 'drone', 'dropper', 'drumstick', 'dumbbell', 'dustpan', 'earplug', 'eclair', 'eel', 'egg_roll', 'electric_chair', 'escargot', 'eyepatch', 'falcon', 'fedora', 'ferret', 'fig_(fruit)', 'file_(tool)', 'first-aid_kit', 'fishbowl', 'flash', 'fleece', 'football_helmet', 'fudge', 'funnel', 'futon', 'gag', 'garbage', 'gargoyle', 'gasmask', 'gemstone', 'generator', 'goldfish', 'gondola_(boat)', 'gorilla', 'gourd', 'gravy_boat', 'griddle', 'grits', 'halter_top', 'hamper', 'hand_glass', 'handcuff', 'handsaw', 'hardback_book', 'harmonium', 'hatbox', 'headset', 'heron', 'hippopotamus', 'hockey_stick', 'hookah', 'hornet', 'hot-air_balloon', 'hotplate', 'hourglass', 'houseboat', 'hummus', 'popsicle', 'ice_pack', 'ice_skate', 'inhaler', 'jelly_bean', 'jewel', 'joystick', 'keg', 'kennel', 'keycard', 'kitchen_table', 'knitting_needle', 'knocker_(on_a_door)', 'koala', 'lab_coat', 'lamb-chop', 'lasagna', 'lawn_mower', 'leather', 'legume', 'lemonade', 'lightning_rod', 'limousine', 'liquor', 'machine_gun', 'mallard', 'mallet', 'mammoth', 'manatee', 'martini', 'mascot', 'masher', 'matchbox', 'microscope', 'milestone', 'milk_can', 'milkshake', 'mint_candy', 'motor_vehicle', 'music_stool', 'nailfile', 'neckerchief', 'nosebag_(for_animals)', 'nutcracker', 'octopus_(food)', 'octopus_(animal)', 'omelet', 'inkpad', 'pan_(metal_container)', 'pantyhose', 'papaya', 'paperback_book', 'paperweight', 'parchment', 'passenger_ship', 'patty_(food)', 'wooden_leg', 'pegboard', 'pencil_box', 'pencil_sharpener', 'pendulum', 'pennant', 'penny_(coin)', 'persimmon', 'phonebook', 'piggy_bank', 'pin_(non_jewelry)', 'ping-pong_ball', 'pinwheel', 'tobacco_pipe', 'pistol', 'pitchfork', 'playpen', 'plow_(farm_equipment)', 'plume', 'pocket_watch', 'poncho', 'pool_table', 'prune', 'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'puncher', 'puppet', 'quesadilla', 'quiche', 'race_car', 'radar', 'rag_doll', 'rat', 'rib_(food)', 'river_boat', 'road_map', 'rodent', 'roller_skate', 'Rollerblade', 'root_beer', 'safety_pin', 'salad_plate', 'salmon_(food)', 'satchel', 'saucepan', 'sawhorse', 'saxophone', 'scarecrow', 'scraper', 'seaplane', 'sharpener', 'Sharpie', 'shaver_(electric)', 'shawl', 'shears', 'shepherd_dog', 'sherbert', 'shot_glass', 'shower_cap', 'shredder_(for_paper)', 'skullcap', 'sling_(bandage)', 'smoothie', 'snake', 'softball', 'sombrero', 'soup_bowl', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)', 'spear', 'crawfish', 'squid_(food)', 'stagecoach', 'steak_knife', 'stepladder', 'stew', 'stirrer', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl', 'sugarcane_(plant)', 'syringe', 'Tabasco_sauce', 'table-tennis_table', 'tachometer', 'taco', 'tambourine', 'army_tank', 'telephoto_lens', 'tequila', 'thimble', 'trampoline', 'trench_coat', 'triangle_(musical_instrument)', 'truffle_(chocolate)', 'vat', 'turnip', 'unicycle', 'vinegar', 'violin', 'vodka', 'vulture', 'waffle_iron', 'walrus', 'wardrobe', 'washbasin', 'water_heater', 'water_gun', 'wolf']

# obtain the conversion from the name to the cate_id
from_cate_name_to_cate_id = {ele['name']:ele['id'] for ele in dataset_content['categories']}
# obtain the base_id and the novel_id
base_cate_id = [from_cate_name_to_cate_id[name] for name in base_cate_name]
novel_cate_id = [from_cate_name_to_cate_id[name] for name in novel_cate_name]

# # aggregate the image id for each categories id
# from_cate_id_to_img_id = {}
# for anno in dataset_content['annotations']:
#     category_id = anno['category_id']
#     image_id = anno['image_id']
#     if category_id not in from_cate_id_to_img_id:
#         from_cate_id_to_img_id[category_id] = []
#     from_cate_id_to_img_id[category_id].append(image_id)

# # remove the repeat image_id 
# for category_id in from_cate_id_to_img_id:
#     from_cate_id_to_img_id[category_id] = list(set(from_cate_id_to_img_id[category_id]))

# # present the number of image with base categories
# all_image_for_base = []
# existed_base_cate = 0
# for category_id in base_cate_id:
#     if category_id in from_cate_id_to_img_id:
#         existed_base_cate += 1
#         all_image_for_base += from_cate_id_to_img_id[category_id]

# all_image_for_base = list(set(all_image_for_base))
# print('image_with_base:', len(all_image_for_base))

# # present the number of image with novel categories
# all_image_for_novel = []
# existed_novel_cate = 0
# for category_id in novel_cate_id:
#     if category_id in from_cate_id_to_img_id:
#         existed_novel_cate += 1
#         all_image_for_novel += from_cate_id_to_img_id[category_id]

# all_image_for_novel = list(set(all_image_for_novel))
# print('image_with_novel:', len(all_image_for_novel))

# # present the base and novel categories in the dataset
# print('existed_base_cate', existed_base_cate, 'existed_novel_cate', existed_novel_cate)

# # aggreate annotation base on the categories and size
# from_cate_id_to_anno_num = {}
# for anno in dataset_content['annotations']:
#     bbox = anno['bbox']
#     category_id = anno['category_id']
#     if category_id not in from_cate_id_to_anno_num:
#         from_cate_id_to_anno_num[category_id] = {'s':0, 'm':0, 'l':0}
#     bbox_size = bbox[3] * bbox[2]
#     if bbox_size < 32 ** 2:
#         from_cate_id_to_anno_num[category_id]['s'] += 1
#     elif bbox_size > 32 ** 2 and bbox_size < 96 ** 2:
#         from_cate_id_to_anno_num[category_id]['m'] += 1
#     else:
#         from_cate_id_to_anno_num[category_id]['l'] += 1

# # aggregate the number of the base intance over different scales
# base_s = 0
# base_m = 0
# base_l = 0
# for category_id in base_cate_id:
#     if category_id in from_cate_id_to_anno_num:
#         base_s += from_cate_id_to_anno_num[category_id]['s']
#         base_m += from_cate_id_to_anno_num[category_id]['m']
#         base_l += from_cate_id_to_anno_num[category_id]['l']

# base_total = base_l + base_m + base_s
# print('base_s:', base_s, 'base_m:', base_m, 'base_l:', base_l, 'base_total:', base_total, 'base_s(precent):', base_s / base_total, 'base_m(precent):', base_m / base_total, 'base_l(precent):', base_l / base_total)

# # aggregate the number of the novel instances over different scales
# novel_s = 0
# novel_m = 0
# novel_l = 0
# for category_id in novel_cate_id:
#     if category_id in from_cate_id_to_anno_num:
#         novel_s += from_cate_id_to_anno_num[category_id]['s']
#         novel_m += from_cate_id_to_anno_num[category_id]['m']
#         novel_l += from_cate_id_to_anno_num[category_id]['l']

# novel_total = novel_s + novel_m + novel_l
# print('novel_s:', novel_s, 'novel_m:', novel_m, 'novel_l:', novel_l, 'novel_total:', novel_total, 'novel_s(precent):', novel_s / novel_total, 'novel_m(precent):', novel_m / novel_total, 'novel_l(precent):', novel_l / novel_total)


# aggregate the annotation base on the image
from_img_id_to_annotation = {}
for anno in dataset_content['annotations']:
    image_id = anno['image_id']
    if image_id not in from_img_id_to_annotation:
        from_img_id_to_annotation[image_id] = {'base':[], 'novel':[]}
    category_id = anno['category_id']
    bbox = anno['bbox']
    bbox_size = bbox[2] * bbox[3]
    if category_id in base_cate_id:
        from_img_id_to_annotation[image_id]['base'].append(bbox_size)
    elif category_id in novel_cate_id:
        from_img_id_to_annotation[image_id]['novel'].append(bbox_size)

# create the from image id to the image info
from_image_id_to_image_size = {}
for anno in dataset_content['images']:
    image_id = anno['id']
    image_size = anno['width'] * anno['height']
    from_image_id_to_image_size[image_id] = image_size

all_base_ratios = []
all_novel_ratios = []
import torch
# aggregate the ratio number
for image_id in from_img_id_to_annotation:
    image_size = from_image_id_to_image_size[image_id]
    novel_bbox_areas = torch.tensor(from_img_id_to_annotation[image_id]['novel'])
    novel_bbox_areas /= image_size
    base_bbox_areas = torch.tensor(from_img_id_to_annotation[image_id]['base'])
    base_bbox_areas /= image_size
    all_base_ratios.append(base_bbox_areas)
    all_novel_ratios.append(novel_bbox_areas)

all_base_ratios = torch.cat(all_base_ratios).tolist()
all_novel_ratios = torch.cat(all_novel_ratios).tolist()
print('all_base_ratios', len(all_base_ratios), 'all_novel_ratios', len(all_novel_ratios))

result = {'base': all_base_ratios, 'novel':all_novel_ratios}

file = open('result.json', 'w')
file.write(json.dumps(result))
file.close()