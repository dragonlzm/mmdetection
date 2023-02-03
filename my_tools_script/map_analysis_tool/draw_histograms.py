# this script aims to draw histogram base on the given data


# calculate the average confidence
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
#file = 'C:\\Users\\Zhuoming Liu\\Desktop\\base_predition_all_valid_confidence_score.pt'
#file = 'C:\\Users\\Zhuoming Liu\\Desktop\\base_predition_all_invalid_confidence_score.pt'

#file = 'C:\\Users\\Zhuoming Liu\\Desktop\\novel_with_trick_predition_all_valid_confidence_score.pt'
#file = 'C:\\Users\\Zhuoming Liu\\Desktop\\novel_with_trick_predition_all_invalid_confidence_score.pt'

#file = 'C:\\Users\\Zhuoming Liu\\Desktop\\novel_predition_all_valid_confidence_score.pt'
#file = 'C:\\Users\\Zhuoming Liu\\Desktop\\novel_predition_all_invalid_confidence_score.pt'
#file = 'C:\\Users\\Zhuoming Liu\\Desktop\\train2017_0_8000_clip_proposal_all_valid_confidence_score.pt'
#file = 'C:\\Users\\Zhuoming Liu\\Desktop\\train2017_0_8000_clip_proposal_all_invalid_confidence_score.pt'
#file = 'C:\\Users\\Zhuoming Liu\\Desktop\\lvis_train_result.json'
file = 'C:\\Users\\Zhuoming Liu\\Desktop\coco_train_result.json'


# res = torch.load(file)
# print(res.shape)
# res = res.tolist()
json_content = json.load(open(file))
#res = json_content['novel']
res = json_content['base']

# matplotlib histogram
plt.hist(res, color = 'blue', edgecolor = 'black', bins = 50)

# seaborn histogram
sns.distplot(res, hist=True, kde=False, 
             bins=50, color = 'blue',
             hist_kws={'edgecolor':'black'})
# Add labels
#plt.title('distribution of bbox size / image size (lvis train base)')
#plt.title('distribution of bbox size / image size (lvis train novel)')
plt.title('distribution of bbox size / image size (coco train base)')
#plt.title('distribution of bbox size / image size (coco train novel)')
#plt.title('distribution of invalid prediction')
plt.xlabel('bbox size / image size')
plt.ylabel('number')
plt.show()