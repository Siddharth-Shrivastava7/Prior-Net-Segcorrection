from dannet_perturb_gt import main, args
# from multiprocessing import Pool
from copy import deepcopy

args_multi = deepcopy(args) 

def main_multi(args_multi):
    # 1. chaning lr only for now
    # 1st run arguments   
    if args_multi.exp== 1:  # lr 
        args1 = deepcopy(args)  
        args1.learning_rate = 0.1 
        args1.gpu_id = '1'
        args1.save_writer_name = 'colored_op_deeplabv3+_' + str(args1.learning_rate)
        main(args= args1) 

    # 2nd run arguments  #lr 
    if args_multi.exp == 2: 
        args2 = deepcopy(args)     
        args2.learning_rate = 1e-2
        args2.gpu_id = '2'
        args2.save_writer_name = 'colored_op_deeplabv3+_' + str(args2.learning_rate)
        main(args= args2)
        
    # 3rd run arguments # lr 
    if args_multi.exp == 3: 
        args3 = deepcopy(args)
        args3.learning_rate = 1e-3
        args3.gpu_id = '3'
        args3.save_writer_name = 'colored_op_deeplabv3+_' + str(args3.learning_rate)
        main(args= args3)

    # 4th run arguments  # batch size changing 
    if args_multi.exp == 4: 
        args4 = deepcopy(args)
        args4.gpu_id = '4'
        args4.batch_size = 8 
        args4.save_writer_name = 'colored_op_deeplabv3+_' + '_bt_' + str(args4.batch_size)
        main(args= args4) 
    
    # 5th run arguments # resnet 50 backbone 
    if args_multi.exp == 5: 
        args5 = deepcopy(args)
        args5.gpu_id = '5'
        args5.model = 'deeplabv3plus_resnet50'
        args5.save_writer_name = 'colored_op_deeplabv3+_' + '_model_' +str(args5.model)
        main(args= args5)
    
    # 6th run arguments # 19 channel mobilenet
    if args_multi.exp == 6: 
        args6 = deepcopy(args)
        args6.gpu_id = '6'
        args6.num_ip_channels = 19
        args6.save_writer_name = 'colored_op_deeplabv3+_' + str(args6.num_ip_channels)
        main(args= args6)

    # 7th run arguments 
    if args_multi.exp == 7: # 19 channel on resnet 18 backbone  
        args7 = deepcopy(args)
        args7.gpu_id = '7'
        args7.num_ip_channels = 19
        args7.model = 'deeplabv3plus_resnet50'
        args7.save_writer_name = 'colored_op_deeplabv3+_' + str(args7.model) + '_' + str(args7.num_ip_channels)
        main(args= args7) 
        
    # 8th run arguments 
    if args_multi.exp == 8: # 19 channel on resnet 18 backbone  
        args8 = deepcopy(args)
        args8.gpu_id = '0'
        args8.num_ip_channels = 19
        args8.model = 'deeplabv3plus_resnet101'
        args8.save_writer_name = 'colored_op_deeplabv3+_' + str(args8.model) + '_' + str(args8.num_ip_channels)
        main(args= args8) 
        
# args_multi = {0: args1, 1: args2, 2: args3}

if __name__ == '__main__':  
    # with Pool(3) as pool: # 3 parallel jobs
    #     results = pool.map(main, args_multi )
    # # for i in range(3):
    #     print(i+1, 'code running')
    #     main(args= args_multi[i])   
    main_multi(args_multi)