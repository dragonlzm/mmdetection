# this script aims to create the validation set which only has base and novel categories
import json

def generate_data_set_base_on_cates(dataset_path, cate_names_list, result_path):
    original_valiation_set = json.load(open(dataset_path))
    # obtain the categories id
    from_name_to_id = {}
    for anno in original_valiation_set['categories']:
        name = anno['name']
        cate_id = anno['id']
        from_name_to_id[name] = cate_id
    all_needed_cates_id = [from_name_to_id[name] for name in cate_names_list]    

    # obtain the annotation
    from_image_id_to_anno = {}
    for anno in original_valiation_set['annotations']:
        cate_id = anno['category_id']
        image_id = anno['image_id']
        if cate_id not in all_needed_cates_id:
            continue
        if image_id not in from_image_id_to_anno:
            from_image_id_to_anno[image_id] = []
        from_image_id_to_anno[image_id].append(anno)

    # obtain the image info
    from_image_id_to_image_anno = {anno['id']:anno for anno in original_valiation_set['images']}

    all_annotations = []
    for image_id in from_image_id_to_anno:
        all_annotations += from_image_id_to_anno[image_id]
    all_image_info = []
    for image_id in from_image_id_to_anno:
        all_image_info.append(from_image_id_to_image_anno[image_id])

    final_res = {'info':original_valiation_set['info'], 
                'licenses':original_valiation_set['licenses'], 
                'images':all_image_info, 
                'annotations':all_annotations, 
                'categories':original_valiation_set['categories']}

    file = open(result_path, 'w')
    file.write(json.dumps(final_res))
    file.close()


if __name__ == "__main__":
    # novel_name = ('airplane', 'bus', 'cat', 'dog', 'cow', 
    #     'elephant', 'umbrella', 'tie', 'snowboard', 
    #     'skateboard', 'cup', 'knife', 'cake', 'couch', 
    #     'keyboard', 'sink', 'scissors')

    # base_name = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
    #             'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
    #             'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
    #             'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
    #             'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
    #             'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
    #             'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
    #             'mouse', 'remote', 'microwave', 'oven', 'toaster', 
    #             'refrigerator', 'book', 'clock', 'vase', 'toothbrush')
    novel_name = ('applesauce', 'apricot', 'arctic_(type_of_shoe)', 'armoire', 'armor', 'ax', 'baboon', 'bagpipe', 'baguet', 'bait', 'ballet_skirt', 'banjo', 'barbell', 'barge', 'bass_horn', 'batter_(food)', 'beachball', 'bedpan', 'beeper', 'beetle', 'Bible', 'birthday_card', 'pirate_flag', 'blimp', 'gameboard', 'bob', 'bolo_tie', 'bonnet', 'bookmark', 'boom_microphone', 'bow_(weapon)', 'pipe_bowl', 'bowling_ball', 'boxing_glove', 'brass_plaque', 'breechcloth', 'broach', 'bubble_gum', 'horse_buggy', 'bulldozer', 'bulletproof_vest', 'burrito', 'cabana', 'locker', 'candy_bar', 'canteen', 'elevator_car', 'car_battery', 'cargo_ship', 'carnation', 'casserole', 'cassette', 'chain_mail', 'chaise_longue', 'chalice', 'chap', 'checkbook', 'checkerboard', 'chessboard', 'chime', 'chinaware', 'poker_chip', 'chocolate_milk', 'chocolate_mousse', 'cider', 'cigar_box', 'clarinet', 'cleat_(for_securing_rope)', 'clementine', 'clippers_(for_plants)', 'cloak', 'clutch_bag', 'cockroach', 'cocoa_(beverage)', 'coil', 'coloring_material', 'combination_lock', 'comic_book', 'compass', 'convertible_(automobile)', 'sofa_bed', 'cooker', 'cooking_utensil', 'corkboard', 'cornbread', 'cornmeal', 'cougar', 'coverall', 'crabmeat', 'crape', 'cream_pitcher', 'crouton', 'crowbar', 'hair_curler', 'curling_iron', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'date_(fruit)', 'detergent', 'diary', 'die', 'dinghy', 'tux', 'dishwasher_detergent', 'diving_board', 'dollar', 'dollhouse', 'dove', 'dragonfly', 'drone', 'dropper', 'drumstick', 'dumbbell', 'dustpan', 'earplug', 'eclair', 'eel', 'egg_roll', 'electric_chair', 'escargot', 'eyepatch', 'falcon', 'fedora', 'ferret', 'fig_(fruit)', 'file_(tool)', 'first-aid_kit', 'fishbowl', 'flash', 'fleece', 'football_helmet', 'fudge', 'funnel', 'futon', 'gag', 'garbage', 'gargoyle', 'gasmask', 'gemstone', 'generator', 'goldfish', 'gondola_(boat)', 'gorilla', 'gourd', 'gravy_boat', 'griddle', 'grits', 'halter_top', 'hamper', 'hand_glass', 'handcuff', 'handsaw', 'hardback_book', 'harmonium', 'hatbox', 'headset', 'heron', 'hippopotamus', 'hockey_stick', 'hookah', 'hornet', 'hot-air_balloon', 'hotplate', 'hourglass', 'houseboat', 'hummus', 'popsicle', 'ice_pack', 'ice_skate', 'inhaler', 'jelly_bean', 'jewel', 'joystick', 'keg', 'kennel', 'keycard', 'kitchen_table', 'knitting_needle', 'knocker_(on_a_door)', 'koala', 'lab_coat', 'lamb-chop', 'lasagna', 'lawn_mower', 'leather', 'legume', 'lemonade', 'lightning_rod', 'limousine', 'liquor', 'machine_gun', 'mallard', 'mallet', 'mammoth', 'manatee', 'martini', 'mascot', 'masher', 'matchbox', 'microscope', 'milestone', 'milk_can', 'milkshake', 'mint_candy', 'motor_vehicle', 'music_stool', 'nailfile', 'neckerchief', 'nosebag_(for_animals)', 'nutcracker', 'octopus_(food)', 'octopus_(animal)', 'omelet', 'inkpad', 'pan_(metal_container)', 'pantyhose', 'papaya', 'paperback_book', 'paperweight', 'parchment', 'passenger_ship', 'patty_(food)', 'wooden_leg', 'pegboard', 'pencil_box', 'pencil_sharpener', 'pendulum', 'pennant', 'penny_(coin)', 'persimmon', 'phonebook', 'piggy_bank', 'pin_(non_jewelry)', 'ping-pong_ball', 'pinwheel', 'tobacco_pipe', 'pistol', 'pitchfork', 'playpen', 'plow_(farm_equipment)', 'plume', 'pocket_watch', 'poncho', 'pool_table', 'prune', 'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'puncher', 'puppet', 'quesadilla', 'quiche', 'race_car', 'radar', 'rag_doll', 'rat', 'rib_(food)', 'river_boat', 'road_map', 'rodent', 'roller_skate', 'Rollerblade', 'root_beer', 'safety_pin', 'salad_plate', 'salmon_(food)', 'satchel', 'saucepan', 'sawhorse', 'saxophone', 'scarecrow', 'scraper', 'seaplane', 'sharpener', 'Sharpie', 'shaver_(electric)', 'shawl', 'shears', 'shepherd_dog', 'sherbert', 'shot_glass', 'shower_cap', 'shredder_(for_paper)', 'skullcap', 'sling_(bandage)', 'smoothie', 'snake', 'softball', 'sombrero', 'soup_bowl', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)', 'spear', 'crawfish', 'squid_(food)', 'stagecoach', 'steak_knife', 'stepladder', 'stew', 'stirrer', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl', 'sugarcane_(plant)', 'syringe', 'Tabasco_sauce', 'table-tennis_table', 'tachometer', 'taco', 'tambourine', 'army_tank', 'telephoto_lens', 'tequila', 'thimble', 'trampoline', 'trench_coat', 'triangle_(musical_instrument)', 'truffle_(chocolate)', 'vat', 'turnip', 'unicycle', 'vinegar', 'violin', 'vodka', 'vulture', 'waffle_iron', 'walrus', 'wardrobe', 'washbasin', 'water_heater', 'water_gun', 'wolf')
    base_name = ('air_conditioner', 'airplane', 'alarm_clock', 'antenna', 'apple', 
           'apron', 'armchair', 'trash_can', 'avocado', 'awning', 'baby_buggy', 
           'backpack', 'handbag', 'suitcase', 'ball', 'balloon', 'banana', 'bandanna', 
           'banner', 'barrel', 'baseball_base', 'baseball', 'baseball_bat', 
           'baseball_cap', 'baseball_glove', 'basket', 'bath_mat', 'bath_towel', 
           'bathtub', 'beanie', 'bear', 'bed', 'bedspread', 'cow', 'beef_(food)', 
           'beer_bottle', 'bell', 'bell_pepper', 'belt', 'belt_buckle', 'bench', 
           'bicycle', 'visor', 'billboard', 'bird', 'birthday_cake', 'blackboard', 
           'blanket', 'blender', 'blinker', 'blouse', 'blueberry', 'boat', 'bolt', 
           'book', 'boot', 'bottle', 'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 
           'box', 'bracelet', 'bread', 'bridal_gown', 'broccoli', 'bucket', 'bun', 
           'buoy', 'bus_(vehicle)', 'butter', 'button', 'cab_(taxi)', 'cabinet', 'cake', 
           'calendar', 'camera', 'can', 'candle', 'candle_holder', 'cap_(headwear)', 
           'bottle_cap', 'car_(automobile)', 'railcar_(part_of_a_train)', 'carrot', 
           'tote_bag', 'cat', 'cauliflower', 'celery', 'cellular_telephone', 'chair', 
           'chandelier', 'choker', 'chopping_board', 'chopstick', 'Christmas_tree', 
           'cigarette', 'cistern', 'clock', 'clock_tower', 'coaster', 'coat', 'coffee_maker', 
           'coffee_table', 'computer_keyboard', 'condiment', 'cone', 'control', 'cookie', 
           'cooler_(for_food)', 'cork_(bottle_plug)', 'edible_corn', 'cowboy_hat', 'crate', 
           'crossbar', 'crumb', 'cucumber', 'cup', 'cupboard', 'cupcake', 'curtain', 'cushion', 
           'deck_chair', 'desk', 'dining_table', 'dish', 'dishtowel', 'dishwasher', 
           'dispenser', 'Dixie_cup', 'dog', 'dog_collar', 'doll', 'doorknob', 'doughnut', 
           'drawer', 'dress', 'dress_suit', 'dresser', 'duck', 'duffel_bag', 'earphone', 
           'earring', 'egg', 'refrigerator', 'elephant', 'fan', 'faucet', 'figurine', 
           'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fireplace', 'fireplug', 
           'fish', 'flag', 'flagpole', 'flip-flop_(sandal)', 'flower_arrangement', 'fork', 
           'frisbee', 'frying_pan', 'giraffe', 'glass_(drink_container)', 'glove', 'goggles', 
           'grape', 'green_bean', 'green_onion', 'grill', 'guitar', 'hairbrush', 'ham', 
           'hair_dryer', 'hand_towel', 'handle', 'hat', 'headband', 'headboard', 'headlight', 
           'helmet', 'hinge', 'home_plate_(baseball)', 'fume_hood', 'hook', 'horse', 'hose', 
           'polar_bear', 'iPod', 'jacket', 'jar', 'jean', 'jersey', 'key', 'kitchen_sink', 
           'kite', 'knee_pad', 'knife', 'knob', 'ladder', 'lamb_(animal)', 'lamp', 'lamppost', 
           'lampshade', 'lanyard', 'laptop_computer', 'latch', 'lemon', 'lettuce', 
           'license_plate', 'life_buoy', 'life_jacket', 'lightbulb', 'lime', 'log', 
           'speaker_(stero_equipment)', 'magazine', 'magnet', 'mailbox_(at_home)', 'manhole', 
           'map', 'marker', 'mask', 'mast', 'mattress', 'microphone', 'microwave_oven', 'milk', 
           'minivan', 'mirror', 'monitor_(computer_equipment) computer_monitor', 'motor', 
           'motor_scooter', 'motorcycle', 'mound_(baseball)', 'mouse_(computer_equipment)', 
           'mousepad', 'mug', 'mushroom', 'napkin', 'necklace', 'necktie', 'newspaper', 
           'notebook', 'nut', 'oar', 'onion', 'orange_(fruit)', 'ottoman', 'oven', 'paddle', 
           'painting', 'pajamas', 'pan_(for_cooking)', 'paper_plate', 'paper_towel', 
           'parking_meter', 'pastry', 'pear', 'pen', 'pencil', 'pepper', 'person', 'piano', 
           'pickle', 'pickup_truck', 'pillow', 'pineapple', 'pipe', 'pitcher_(vessel_for_liquid)', 
           'pizza', 'place_mat', 'plate', 'pole', 'polo_shirt', 'pop_(soda)', 'poster', 'pot', 
           'flowerpot', 'potato', 'printer', 'propeller', 'quilt', 'radiator', 'rearview_mirror', 
           'reflector', 'remote_control', 'ring', 'rubber_band', 'plastic_bag', 
           'saddle_(on_an_animal)', 'saddle_blanket', 'sail', 'salad', 'saltshaker', 
           'sandal_(type_of_shoe)', 'sandwich', 'saucer', 'sausage', 
           'scale_(measuring_instrument)', 'scarf', 'scissors', 'scoreboard', 'scrubbing_brush', 
           'sheep', 'shirt', 'shoe', 'shopping_bag', 'short_pants', 'shoulder_bag', 
           'shower_head', 'shower_curtain', 'signboard', 'sink', 'skateboard', 'ski', 
           'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'snowboard', 'soap', 'soccer_ball', 
           'sock', 'sofa', 'soup', 'spatula', 'spectacles', 'spoon', 'statue_(sculpture)', 
           'steering_wheel', 'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strap', 
           'straw_(for_drinking)', 'strawberry', 'street_sign', 'streetlight', 
           'suit_(clothing)', 'sunglasses', 'surfboard', 'sweater', 'sweatshirt', 
           'swimsuit', 'table', 'tablecloth', 'tag', 'taillight', 'tank_(storage_vessel)', 
           'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tarp', 'teapot', 
           'teddy_bear', 'telephone', 'telephone_pole', 'television_set', 'tennis_ball', 
           'tennis_racket', 'thermostat', 'tinfoil', 'tissue_paper', 'toaster', 'toaster_oven', 
           'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toothbrush', 'toothpaste', 
           'toothpick', 'cover', 'towel', 'towel_rack', 'toy', 'traffic_light', 
           'trailer_truck', 'train_(railroad_vehicle)', 'tray', 'tripod', 'trousers', 'truck', 
           'umbrella', 'underwear', 'urinal', 'vase', 'vent', 'vest', 'wall_socket', 'wallet', 
           'watch', 'water_bottle', 'watermelon', 'weathervane', 'wet_suit', 'wheel', 
           'windshield_wiper', 'wine_bottle', 'wineglass', 'blinder_(for_horses)', 'wristband', 
           'wristlet', 'zebra', 'aerosol_can', 'alcohol', 'alligator', 'almond', 'ambulance', 
           'amplifier', 'anklet', 'aquarium', 'armband', 'artichoke', 'ashtray', 'asparagus', 
           'atomizer', 'award', 'basketball_backboard', 'bagel', 'bamboo', 'Band_Aid', 
           'bandage', 'barrette', 'barrow', 'basketball', 'bat_(animal)', 'bathrobe', 
           'battery', 'bead', 'bean_curd', 'beanbag', 'beer_can', 'beret', 'bib', 'binder', 
           'binoculars', 'birdfeeder', 'birdbath', 'birdcage', 'birdhouse', 'black_sheep', 
           'blackberry', 'blazer', 'bobbin', 'bobby_pin', 'boiled_egg', 'deadbolt', 'bookcase', 
           'booklet', 'bottle_opener', 'bouquet', 'bowler_hat', 'suspenders', 'brassiere', 
           'bread-bin', 'briefcase', 'broom', 'brownie', 'brussels_sprouts', 'bull', 'bulldog', 
           'bullet_train', 'bulletin_board', 'bullhorn', 'bunk_bed', 'business_card', 
           'butterfly', 'cabin_car', 'calculator', 'calf', 'camcorder', 'camel', 'camera_lens', 
           'camper_(vehicle)', 'can_opener', 'candy_cane', 'walking_cane', 'canister', 'canoe', 
           'cantaloup', 'cape', 'cappuccino', 'identity_card', 'card', 'cardigan', 
           'horse_carriage', 'cart', 'carton', 'cash_register', 'cast', 'cayenne_(spice)', 
           'CD_player', 'cherry', 'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 
           'crisp_(potato_chip)', 'chocolate_bar', 'chocolate_cake', 'slide', 'cigarette_case', 
           'clasp', 'cleansing_agent', 'clip', 'clipboard', 'clothes_hamper', 'clothespin', 
           'coat_hanger', 'coatrack', 'cock', 'coconut', 'coffeepot', 'coin', 'colander', 
           'coleslaw', 'pacifier', 'corkscrew', 'cornet', 'cornice', 'corset', 'costume', 
           'cowbell', 'crab_(animal)', 'cracker', 'crayon', 'crescent_roll', 'crib', 
           'crock_pot', 'crow', 'crown', 'crucifix', 'cruise_ship', 'police_cruiser', 'crutch', 
           'cub_(animal)', 'cube', 'cufflink', 'trophy_cup', 'dartboard', 'deer', 
           'dental_floss', 'diaper', 'dish_antenna', 'dishrag', 'dolphin', 'domestic_ass', 
           'doormat', 'underdrawers', 'dress_hat', 'drill', 'drum_(musical_instrument)', 
           'duckling', 'duct_tape', 'dumpster', 'eagle', 'easel', 'egg_yolk', 'eggbeater', 
           'eggplant', 'elk', 'envelope', 'eraser', 'Ferris_wheel', 'ferry', 'fighter_jet', 
           'file_cabinet', 'fire_hose', 'fish_(food)', 'fishing_rod', 'flamingo', 'flannel', 
           'flap', 'flashlight', 'flipper_(footwear)', 'flute_glass', 'foal', 'folding_chair', 
           'food_processor', 'football_(American)', 'footstool', 'forklift', 'freight_car', 
           'French_toast', 'freshener', 'frog', 'fruit_juice', 'garbage_truck', 'garden_hose', 
           'gargle', 'garlic', 'gazelle', 'gelatin', 'giant_panda', 'gift_wrap', 'ginger', 
           'cincture', 'globe', 'goat', 'golf_club', 'golfcart', 'goose', 'grater', 
           'gravestone', 'grizzly', 'grocery_bag', 'gull', 'gun', 'hairnet', 'hairpin', 
           'hamburger', 'hammer', 'hammock', 'hamster', 'handcart', 'handkerchief', 'veil', 
           'headscarf', 'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'highchair', 
           'hog', 'honey', 'hot_sauce', 'hummingbird', 'icecream', 'ice_maker', 'igniter', 
           'iron_(for_clothing)', 'ironing_board', 'jam', 'jeep', 'jet_plane', 'jewelry', 
           'jumpsuit', 'kayak', 'kettle', 'kilt', 'kimono', 'kitten', 'kiwi_fruit', 'ladle', 
           'ladybug', 'lantern', 'legging_(clothing)', 'Lego', 'lion', 'lip_balm', 'lizard', 
           'lollipop', 'loveseat', 'mail_slot', 'mandarin_orange', 'manger', 'mashed_potato', 
           'mat_(gym_equipment)', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine', 
           'melon', 'mitten', 'mixer_(kitchen_tool)', 'money', 'monkey', 'muffin', 
           'musical_instrument', 'needle', 'nest', 'newsstand', 'nightshirt', 
           'noseband_(for_animals)', 'notepad', 'oil_lamp', 'olive_oil', 'orange_juice', 
           'ostrich', 'overalls_(clothing)', 'owl', 'packet', 'pad', 'padlock', 'paintbrush', 
           'palette', 'pancake', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol', 
           'parka', 'parrot', 'passenger_car_(part_of_a_train)', 'passport', 'pea_(food)', 
           'peach', 'peanut_butter', 'peeler_(tool_for_fruit_and_vegetables)', 'pelican', 
           'penguin', 'pepper_mill', 'perfume', 'pet', 'pew_(church_bench)', 
           'phonograph_record', 'pie', 'pigeon', 'pinecone', 'pita_(bread)', 'platter', 
           'pliers', 'pocketknife', 'poker_(fire_stirring_tool)', 'pony', 'postbox_(public)', 
           'postcard', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel', 
           'projectile_(weapon)', 'projector', 'pumpkin', 'puppy', 'rabbit', 'racket', 
           'radio_receiver', 'radish', 'raft', 'raincoat', 'ram_(animal)', 'raspberry', 
           'razorblade', 'reamer_(juicer)', 'receipt', 'recliner', 'record_player', 
           'rhinoceros', 'rifle', 'robe', 'rocking_chair', 'rolling_pin', 
           'router_(computer_equipment)', 'runner_(carpet)', 'saddlebag', 'salami', 
           'salmon_(fish)', 'salsa', 'school_bus', 'screwdriver', 'sculpture', 'seabird', 
           'seahorse', 'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark', 
           'shaving_cream', 'shield', 'shopping_cart', 'shovel', 'silo', 'skewer', 'sled', 
           'sleeping_bag', 'slipper_(footwear)', 'snowman', 'snowmobile', 'solar_array', 
           'soupspoon', 'sour_cream', 'spice_rack', 'spider', 'sponge', 'sportswear', 
           'spotlight', 'squirrel', 'stapler_(stapling_machine)', 'starfish', 'steak_(food)', 
           'step_stool', 'stereo_(sound_system)', 'strainer', 'sunflower', 'sunhat', 'sushi', 
           'mop', 'sweat_pants', 'sweatband', 'sweet_potato', 'sword', 'table_lamp', 
           'tape_measure', 'tapestry', 'tartan', 'tassel', 'tea_bag', 'teacup', 'teakettle', 
           'telephone_booth', 'television_camera', 'thermometer', 'thermos_bottle', 'thread', 
           'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinsel', 
           'toast_(food)', 'toolbox', 'tortilla', 'tow_truck', 'tractor_(farm_equipment)', 
           'dirt_bike', 'tricycle', 'trunk', 'turban', 'turkey_(food)', 'turtle', 
           'turtleneck_(clothing)', 'typewriter', 'urn', 'vacuum_cleaner', 'vending_machine', 
           'videotape', 'volleyball', 'waffle', 'wagon', 'wagon_wheel', 'walking_stick', 
           'wall_clock', 'automatic_washer', 'water_cooler', 'water_faucet', 'water_jug', 
           'water_scooter', 'water_ski', 'water_tower', 'watering_can', 'webcam',
           'wedding_cake', 'wedding_ring', 'wheelchair', 'whipped_cream', 'whistle', 
           'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)', 'windsock', 
           'wine_bucket', 'wok', 'wooden_spoon', 'wreath', 'wrench', 'yacht', 'yogurt', 
           'yoke_(animal_equipment)', 'zucchini')

    #dataset_path = 'data/coco/annotations/instances_train2017_0_8000.json'
    #result_path = 'data/coco/annotations/instances_train2017_0_8000_novel17.json'
    dataset_path = '/data/zhuoming/detection/lvis_v1/annotations/lvis_v1_val.json'
    #result_path = '/data/zhuoming/detection/lvis_v1/annotations/lvis_v1_val_novel.json'    
    #generate_data_set_base_on_cates(dataset_path, novel_name, result_path)
    
    result_path = '/data/zhuoming/detection/lvis_v1/annotations/lvis_v1_val_base.json'
    generate_data_set_base_on_cates(dataset_path, base_name, result_path)
