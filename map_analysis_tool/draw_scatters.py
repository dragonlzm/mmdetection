import numpy as np
import matplotlib.pyplot as plt
import torch

x_file = 'C:\\Users\\Zhuoming Liu\\Desktop\\train2017_0_8000_clip_proposal_all_valid_confidence_iop_or_iog.pt'
y_file = 'C:\\Users\\Zhuoming Liu\\Desktop\\train2017_0_8000_clip_proposal_all_valid_confidence_score.pt'

x_invalid_file = 'C:\\Users\\Zhuoming Liu\\Desktop\\train2017_0_8000_clip_proposal_all_invalid_confidence_iop_or_iog.pt'
y_invalid_file = 'C:\\Users\\Zhuoming Liu\\Desktop\\train2017_0_8000_clip_proposal_all_invalid_confidence_score.pt'


x_value = torch.cat([torch.load(x_file), torch.load(x_invalid_file)], dim=0).numpy()
y_value = torch.cat([torch.load(y_file), torch.load(y_invalid_file)], dim=0).numpy()

print(x_value.shape)

rand_idx = np.random.choice(x_value.shape[0], 1000)
# x_value = x_value[rand_idx]
# y_value = y_value[rand_idx]


#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.scatter(x_value, y_value, s=1, c='blue', alpha=0.5)
plt.title('distribution of valid prediction, conf vs max(iop, iog)')
#plt.title('distribution of invalid prediction')
plt.xlabel('max(iop, iog)')
plt.ylabel('confidence score')
plt.show()