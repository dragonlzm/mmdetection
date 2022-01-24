# go through all the image in the dict:
for count_i, image_id in enumerate(from_img_id_to_bbox.keys()):
    #filenname = from_img_id_to_bbox[image_id]['path']
    
    #load the image and convert to numpy
    #img_bytes = file_client.get(filenname)
    #img = mmcv.imfrombytes(img_bytes, flag='color', channel_order='rgb')
    
    #h_patch_num = 16
    #w_patch_num = 16
    h_patch_num = 8
    w_patch_num = 8
    #h_patch_num = 4
    #w_patch_num = 4
    
    #H1, W1, channel1 = img.shape
    H, W, channel = from_img_id_to_bbox[image_id]['shape']

    patch_H, patch_W = H / h_patch_num, W / w_patch_num
    h_pos = [int(patch_H) * i for i in range(h_patch_num + 1)]
    w_pos = [int(patch_W) * i for i in range(w_patch_num + 1)]

    # the grid match list
    match_list = [[[] for j in range(w_patch_num)] for i in range(h_patch_num)]

    for bbox in from_img_id_to_bbox[image_id]['bbox']:
        # for each bbox we need to calculate whether the bbox is inside the grid
        x, y, w, h, cat_id = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
        x_idx = math.floor(x / patch_W)
        y_idx = math.floor(y / patch_H)
        x_end_pos = w_pos[x_idx+1]
        y_end_pos = h_pos[y_idx+1]
        br_x = x + w
        br_y = y + h
        # if the bbox is inside the grid
        # assign the bbox to the grid and save the categories
        if br_x <= x_end_pos and br_y <= y_end_pos:
            match_list[y_idx][x_idx].append([x, y, br_x, br_y, bbox[4]])
            continue
        # then we calculate the iou 
        # edge condition
        if br_x == math.floor(br_x / patch_W) * patch_W:
            br_x_idx = math.floor(br_x / patch_W) - 1
        else:
            br_x_idx = math.floor(br_x / patch_W)

        if br_y == math.floor(br_y / patch_H) * patch_H:
            br_y_idx = math.floor(br_y / patch_H) - 1
        else:    
            br_y_idx = math.floor(br_y / patch_H)

        # prepare the tensor for calculate the iou
        #the_gt_tensor = torch.tensor([[[x, y, br_x, br_y]]])
        # prepare the tensor for the grid
        #the_grid_list = []
        #for now_x_idx in range(x_idx, br_x_idx+1):
        #    for now_y_idx in range(y_idx, br_y_idx+1):
        #        now_grid = torch.tensor([[[w_pos[now_x_idx], h_pos[now_y_idx], w_pos[now_x_idx+1], h_pos[now_y_idx+1]]]])
        #        the_grid_list.append(now_grid)
        #the_grid_tensor = torch.cat(the_grid_list, dim=1)
        #assign_result = bbox_overlaps(the_gt_tensor, the_grid_tensor)
        #assign_result = bbox_overlaps(the_gt_tensor, the_grid_tensor, mode='iof')
        #result = assign_result >= 0.5
        #result = (assign_result == torch.max(assign_result))
        #print(assign_result)
        #break
        for now_x_idx in range(x_idx, br_x_idx+1):
            for now_y_idx in range(y_idx, br_y_idx+1):
                #list_idx = (now_y_idx - y_idx) * (br_x_idx+1 - x_idx) + (now_x_idx - x_idx)
                #if result[0][0][list_idx] == True:
                match_list[now_y_idx][now_x_idx].append([x, y, br_x, br_y, bbox[4]])

    if count_i % 1000 == 0:
        print(count_i)

    # the post-processing the final result should be a 8*8 tensor which each element is the categories id 
    final_assign_tensor = torch.zeros(h_patch_num, w_patch_num)
    for i in range(h_patch_num):
        for j in range(w_patch_num):
            if len(match_list[i][j]) == 0:
                final_assign_tensor[i][j] = 10000
            elif len(match_list[i][j]) == 1:
                final_assign_tensor[i][j] = match_list[i][j][0][-1]
            else:
                # the grid tensor 
                the_grid_tensor = torch.tensor([[[w_pos[j], h_pos[i], w_pos[j+1], h_pos[i+1]]]])
                # the possible match tensor
                all_tensor_list = []
                for bbox in match_list[i][j]:
                    now_tensor = torch.tensor([[bbox[:-1]]])
                    all_tensor_list.append(now_tensor)
                the_gt_tensor = torch.cat(all_tensor_list, dim=1)
                
                #calculate the iou between the gt bboxes and the 
                iou_value = bbox_overlaps(the_grid_tensor, the_gt_tensor, mode='iof')

                final_assign_tensor[i][j] = match_list[i][j][torch.argmax(iou_value)][-1]
    
    all_assigned_res.append(torch.unsqueeze(final_assign_tensor, dim=0))
    #all_assigned_res[image_id] = final_assign_tensor

    # crop the image and combine them together
    #result = []
    #for i in range(h_patch_num):
    #    h_start_pos = h_pos[i]
    #    h_end_pos = h_pos[i+1]
    #    for j in range(w_patch_num):
    #        w_start_pos = w_pos[j]
    #        w_end_pos = w_pos[j+1]
            # cropping the img into the patches which size is (H/8) * (W/8)
            # use the numpy to crop the image
    #        now_patch = img[h_start_pos: h_end_pos, w_start_pos: w_end_pos, :]
    #        PIL_image = Image.fromarray(np.uint8(now_patch))
            # do the preprocessing
    #        new_patch = preprocess(PIL_image)
    #        result.append(new_patch.unsqueeze(dim=0))

    #cropped_patches = torch.cat(result, dim=0).cuda()
    #with torch.no_grad():
    #    image_features = model.encode_image(cropped_patches)
    #all_feature_res.append(torch.unsqueeze(image_features, dim=0).cpu())
