import argparse
import os,sys
import fundus_eval
"""
cnn_model --fundus-- 0
                     |__ckpt
                     |__meta.graph  
                     1
                     2
                     3

"""

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--path_dir' , help='dir to saved data') # e.g ./paths/cropped_original_fundus_300x300/
    parser.add_argument('--model_dir', help='dir to saved data')  # e.g ./cnn_model/fundus/

    args = parser.parse_args()
    folder_path, subfolder_names , subfolder_files=os.walk(args.model_dir).next()
    if not os.path.isdir('./cnn_models/ensemble'):
        os.mkdir('./cnn_models/ensemble')
    tmp_folder_path=os.path.join('./cnn_models/ensemble', args.path_dir.split('/')[-1]) #e.g ./ensemble/cropped_original_fundus_300x300/
    if not os.path.isdir(tmp_folder_path):
        os.mkdir(os.path.join('./cnn_models/ensemble', args.path_dir.split('/')[-1]))
        print 'folder created :  ',os.path.join('./cnn_models/ensemble', args.path_dir.split('/')[-1])
        # e.g ./ensemble/cropped_original_fundus_300x300/
    for subfolder_name in subfolder_names:
        target_model_folder=os.path.join(tmp_folder_path  , subfolder_name) #./cnn_model/fundus/1/
        acc_list , pred_list = fundus_eval.eval_from_numpy_image(path_dir= args.path_dir , model_dir=target_model_folder)






    print 'the number of model' ,len(subfolder_files)


