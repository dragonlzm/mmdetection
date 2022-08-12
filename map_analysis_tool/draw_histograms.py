# this script aims to draw histogram base on the given data


# calculate the average confidence
import torch
import matplotlib.pyplot as plt
import seaborn as sns
#file = 'C:\\Users\\Zhuoming Liu\\Desktop\\base_predition_all_valid_confidence_score.pt'
#file = 'C:\\Users\\Zhuoming Liu\\Desktop\\base_predition_all_invalid_confidence_score.pt'

#file = 'C:\\Users\\Zhuoming Liu\\Desktop\\novel_with_trick_predition_all_valid_confidence_score.pt'
#file = 'C:\\Users\\Zhuoming Liu\\Desktop\\novel_with_trick_predition_all_invalid_confidence_score.pt'

#file = 'C:\\Users\\Zhuoming Liu\\Desktop\\novel_predition_all_valid_confidence_score.pt'
#file = 'C:\\Users\\Zhuoming Liu\\Desktop\\novel_predition_all_invalid_confidence_score.pt'
file = 'C:\\Users\\Zhuoming Liu\\Desktop\\train2017_0_8000_clip_proposal_all_valid_confidence_score.pt'
#file = 'C:\\Users\\Zhuoming Liu\\Desktop\\train2017_0_8000_clip_proposal_all_invalid_confidence_score.pt'

res = torch.load(file)
print(res.shape)
res = res.tolist()

# matplotlib histogram
plt.hist(res, color = 'blue', edgecolor = 'black',
         bins = 20)

# seaborn histogram
sns.distplot(res, hist=True, kde=False, 
             bins=20, color = 'blue',
             hist_kws={'edgecolor':'black'})
# Add labels
plt.title('distribution of valid prediction')
#plt.title('distribution of invalid prediction')
plt.xlabel('confidence score')
plt.ylabel('number')
plt.show()