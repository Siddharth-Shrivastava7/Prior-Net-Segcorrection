import numpy as np

## discrete single pixels 
def perturb_pixels(label, num_pix_perturb): 
    assert type(label) is np.ndarray 
    
    label_perturb = np.copy(label) 
    for _ in range(num_pix_perturb): 
        randx, randy = np.random.randint(label_perturb.shape[0]), np.random.randint(label_perturb.shape[1])   
        actual_label = label_perturb[randx, randy]
        while actual_label!=255:
            perturb_label = np.random.randint(19)  
            if actual_label!= perturb_label: break  
        if actual_label == 255: 
            perturb_label = 255 
        label_perturb[randx,randy] = perturb_label 
    
    return label_perturb 


## square for now 
def perturb_discrete_patches(label, num_patch_perturb, patch_size): 
    assert type(label) is np.ndarray 
    label_perturb = np.copy(label) 

    for _ in range(num_patch_perturb):  
        randy, randx = np.random.randint(label_perturb.shape[0] - patch_size), np.random.randint(label_perturb.shape[1] - patch_size)    

        lp_patch = label_perturb[randy:randy+ patch_size, randx:randx+ patch_size]  
        
        for y in range(lp_patch.shape[0]): 
            for x in range(lp_patch.shape[1]):
                actual_label = lp_patch[y,x] 
                while actual_label!=255:
                    perturb_label = np.random.randint(19)  
                    if actual_label!= perturb_label: break  
                if actual_label == 255: 
                    perturb_label = 255 
                lp_patch[y,x] = perturb_label
        
        label_perturb[randy:randy+ patch_size, randx:randx+ patch_size] = lp_patch  

    return label_perturb
