import numpy as np
import pandas as pd
import random
from random import shuffle
from collections import Counter

pd.options.display.float_format = '{:.2f}'.format

dep_ad_6 =  [[ 0.3397488 ],
 [-6.182128  ],
 [-2.6109889 ],
 [-2.5829592 ],
 [-1.033362  ],
 [-3.050375  ],
 [-0.7875966 ],
 [-0.5950744 ],
 [-0.39100784],
 [-1.9569271 ],
 [-0.70979524],
 [-3.7272725 ],
 [-1.9902912 ],
 [-3.3369398 ],
 [-2.253963  ],
 [ 0.02786294],
 [-2.7981768 ],
 [-0.8753541 ],
 [ 0.70363784],
 [-1.3100684 ],
 [-0.6046724 ],
 [ 1.4784343 ],
 [-0.93942064],
 [ 1.0404972 ],
 [-6.530124  ],
 [-0.6716486 ],
 [ 0.36578766],
 [ 0.27929026],
 [-0.84329766],
 [-1.683871  ],
 [ 0.39861476],
 [ 0.12859401],
 [-0.4600977 ],
 [-1.0405974 ],
 [-1.9248775 ],
 [ 0.15660389],
 [ 0.53004074],
 [-3.6520119 ],
 [ 0.6518822 ],
 [-1.3203028 ],
 [-1.9569259 ],
 [-0.98439014],
 [ 0.5866548 ],
 [-0.40156528],
 [-0.39172035],
 [ 0.34279835],
 [-3.963763  ],
 [ 1.7069796 ],
 [-0.05377274],
 [ 0.20007671],
 [-4.6278877 ],
 [ 0.28723565],
 [-0.47016105],
 [-0.13696007],
 [-2.3982604 ],
 [-0.03026873],
 [-1.1243658 ],
 [-0.6353848 ],
 [-0.26162493],
 [-0.6401419 ],
 [ 1.3319451 ],
 [-0.8466409 ],
 [-0.68166554],
 [ 0.8023891 ],
 [ 0.22485404],
 [-1.6829535 ],
 [ 1.605563  ],
 [-0.2619983 ],
 [ 0.22109857],
 [-0.5094976 ],
 [-0.8589231 ],
 [ 1.8771286 ],
 [-1.194045  ],
 [-1.68305   ],
 [-0.6809345 ],
 [-0.7698768 ],
 [-0.05977101],
 [-1.799577  ],
 [-0.26304343],
 [-2.401426  ],
 [-0.69815916],
 [-2.8420043 ],
 [-3.7486377 ],
 [-1.8262649 ],
 [-0.5231309 ],
 [-0.6010335 ],
 [-0.57270694],
 [-0.15428193],
 [-0.46923572],
 [ 0.2668033 ],
 [-0.8216184 ],
 [-3.0976593 ],
 [ 0.60603094],
 [-1.6606001 ],
 [-0.80263776],
 [-1.4099039 ],
 [-0.71217465],
 [ 0.15501687],
 [-2.7341068 ],
 [-0.7166905 ]]


dep_ad_9 =  [[ 0.74082696],
 [ 1.2213591 ],
 [ 1.3755003 ],
 [-2.1187363 ],
 [ 0.4146689 ],
 [ 0.5954165 ],
 [ 0.31109452],
 [ 0.16466627],
 [-3.1116116 ],
 [-2.4775698 ],
 [ 0.09512657],
 [-2.4818335 ],
 [-1.6943353 ],
 [ 0.0662456 ],
 [-0.15185003],
 [-2.0015154 ],
 [-3.6018267 ],
 [-0.6174673 ],
 [-4.454662  ],
 [-1.4582055 ],
 [-0.7836857 ],
 [-0.52876306],
 [-2.2666326 ],
 [-1.3851441 ],
 [-1.006258  ],
 [ 0.78977835],
 [-0.2852271 ],
 [ 0.08895236],
 [-0.66482425],
 [-0.03457318],
 [ 0.3699285 ],
 [-1.9360262 ],
 [-4.0889244 ],
 [-0.36789584],
 [-0.4572267 ],
 [ 0.7804881 ],
 [-0.84384394],
 [-2.6217563 ],
 [-1.540374  ],
 [ 0.01228092],
 [-2.8032405 ],
 [-2.107087  ],
 [ 2.2811265 ],
 [-0.37784234],
 [ 1.0996221 ],
 [-0.11514137],
 [-2.6737158 ],
 [-4.687642  ],
 [ 0.11948603],
 [-2.2229064 ],
 [-0.78003466],
 [-1.5227368 ],
 [-0.76997745],
 [-1.2344922 ],
 [-0.8337941 ],
 [ 0.4642086 ],
 [-0.8630465 ],
 [-0.36583024],
 [-3.5570354 ],
 [-2.9033246 ],
 [-0.31445178],
 [-0.94588625],
 [-0.05996021],
 [-0.8337299 ],
 [ 0.12597995],
 [ 0.16862868],
 [-0.4740412 ],
 [-1.1398344 ],
 [ 0.74776715],
 [ 0.45456514],
 [-1.9093713 ],
 [-0.6198179 ],
 [ 0.2306173 ],
 [-1.431161  ],
 [-3.1495552 ],
 [-0.991905  ],
 [ 0.24985167],
 [-3.2972634 ],
 [-0.3670874 ],
 [-0.37091452],
 [-0.36459187],
 [ 0.7052915 ],
 [-1.9329863 ],
 [-0.80129623],
 [-1.9975928 ],
 [-1.0179666 ],
 [-2.7698574 ],
 [-1.0865781 ],
 [-2.70768   ],
 [-0.68842286],
 [-3.850188  ],
 [-1.5086308 ],
 [ 0.7842929 ],
 [-1.8169643 ],
 [ 1.7969111 ],
 [-1.2395781 ],
 [-3.4364488 ],
 [ 1.2734022 ],
 [-2.7803066 ],
 [ 0.03487867]]

weighted_group = np.concatenate((dep_ad_9, dep_ad_6), axis=1)
print(weighted_group)
import matplotlib.pyplot as plt

# Creating dataset
np.random.seed(10)

fig, ax= plt.subplots()

main=ax.boxplot(weighted_group, patch_artist=True)
m = weighted_group.mean(axis=0)
st = weighted_group.std(axis=0)
ax.set_title('Ad-Depression: Q-learning rewards for different States')
for i, line in enumerate(main['medians']):
    x, y = line.get_xydata()[1]
    text = ' μ={:.2f}\n σ={:.2f}'.format(m[i], st[i])
    ax.annotate(text, xy=(x, y))
ax.set_xticks([1, 2])
ax.set_xticklabels(["9 states for samples>50","6 states for samples>100"], rotation=10)
# ax.set_ylim(-20,10)

colors=["forestgreen", "blue"]
 
for patch, color in zip(main['boxes'],colors):
    patch.set_facecolor(color)

 

plt.savefig('ad_dep.png')
# show plot
plt.show()