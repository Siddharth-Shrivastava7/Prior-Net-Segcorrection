from dannet_perturb_gt import main, args
# from multiprocessing import Pool
from copy import deepcopy

args_multi = deepcopy(args) 

def main_multi(args_multi):
    # 1st run arguments   
    if args_multi.exp== 1:   
        args1 = deepcopy(args)  
        args1.synthetic_perturb = 'synthetic_manual_dannet_20n_100p_1024im_synthetic' 
        args1.gpu_id = '1'
        args1.img_size = 1024
        args1.save_writer_name = 'colored_deeplabv3+_1024_synthetic_color' 
        main(args= args1) 

    # 2nd run arguments 
    if args_multi.exp == 2: 
        args2 = deepcopy(args)  
        args2.synthetic_perturb = 'synthetic_manual_dannet_20n_100p_1024im' 
        args2.gpu_id = '2'
        args2.img_size = 1024
        args2.restore_from_color_mapping = 'None'
        args2.save_writer_name = 'colored_deeplabv3+_1024' 
        main(args= args2)
        
    # 3rd run arguments  
    if args_multi.exp == 3: 
        args3 = deepcopy(args)
        args3.weighted_ce = True 
        args3.naive_weighting = True
        args3.gpu_id = '3'
        args3.save_writer_name = 'colored_deeplabv3+_weighted_naively'
        main(args= args3)

    # 4th run arguments  # batch size changing 
    if args_multi.exp == 4: 
        args4 = deepcopy(args)
        args4.weighted_ce = True 
        args4.dannet_acdc_weighting = True
        args4.gpu_id = '4'
        args4.save_writer_name = 'colored_deeplabv3+_dannet_weighting'
        main(args= args4) 
    
    if args_multi.exp == 5: 
        args5 = deepcopy(args)
        args5.batch_size = 2
        args5.gpu_id = '5'
        args5.save_writer_name = 'colored_deeplabv3+_batch_2'
        main(args= args5) 
    
    if args_multi.exp == 6: 
        args6 = deepcopy(args)
        args6.batch_size = 1
        args6.gpu_id = '6'
        args6.save_writer_name = 'colored_deeplabv3+_batch_1'
        main(args= args6) 
    
if __name__ == '__main__':  
    main_multi(args_multi)