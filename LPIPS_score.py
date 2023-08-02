import os
import torch
import pdb
import lpips
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

loss_fn_alex = lpips.LPIPS(net='alex')

tfs = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


root = "result/BDD_OOD/" 
truth_dir = root + "truth"
inpaint_dir = root + "out"
result_dir = root +"result"
scores_dict = {}
OOD_score = []
ID_score = []
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

for img_name in os.listdir(truth_dir):

    print(truth_dir+"/"+img_name)
    
    # Read image
    img_pillow_original = Image.open(truth_dir+"/"+img_name).convert('RGB')
    img_original = tfs(img_pillow_original)
    img_pillow_impaint = Image.open(inpaint_dir+"/"+img_name).convert('RGB')
    img_inpainted = tfs(img_pillow_impaint)
    
    # Compute Score
    score = loss_fn_alex(img_original, img_inpainted).item()
    print(f"OOD Score: {score}")
    scores_dict.update({img_name: OOD_score})
    if "OOD" in img_name:
        OOD_score.append(score)
    else: 
        ID_score.append(score)
    
    # Visualization
    fig, axs = plt.subplots(figsize=(25, 25), nrows=1, ncols=2, constrained_layout=True)
    axs = axs.flatten()
    axs[0].set_title('Original Image', fontsize=40, color = 'purple',fontweight="bold", pad = 60)
    axs[1].set_title('Inpainted Image', fontsize=40, color = 'purple',fontweight="bold", pad = 60)
    axs[0].imshow(img_pillow_original)
    axs[1].imshow(img_pillow_impaint)
    axs[0].axis('off')
    axs[1].axis('off')
    plt.figtext(0.5, 0.88, f'{img_name[:-4]}' , fontsize=55, color = 'blue',fontweight="bold", ha='center', va='center_baseline')
    plt.figtext(0.5, 0.83, f'OOD Score: {round(score, 5)}', fontsize=50, color = 'green',fontweight="bold", ha='center', va='center_baseline')
    plt.savefig(result_dir + "/" +img_name, bbox_inches='tight', dpi=400)  
    plt.close()
    
np.save(result_dir + "/" + "score_dict", scores_dict)
np.save(result_dir + "/" + "ID_scores", ID_score)
np.save(result_dir + "/" + "OOD_scores", OOD_score)

