import argparse
import os,sys
import fundus_eval
import pickle
"""
cnn_model --fundus-- 0
                     |__ckpt
                     |__meta.graph  
                     1
                     2
                     3

ensemble--result_saved_folder--target_folder
    

"""


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--path_dir' , help='dir to saved data') # e.g ./paths/cropped_original_fundus_300x300/2/
    parser.add_argument('--model_dir', help='dir to saved data')  # e.g ./cnn_model/fundus/
    args = parser.parse_args()

    folder_path, subfolder_names , subfolder_files=os.walk(args.model_dir).next()
    if not os.path.isdir('./ensemble'):
        os.mkdir('./ensemble')
        print 'ensemble folder was created'
    ensemble_save_folder=os.path.join('./ensemble', args.path_dir.split('/')[-3]) #e.g ensemble_save_path= ./ensemble/cropped_original_fundus_300x300
    print 'folder name saved model',subfolder_names
    print  args.path_dir.split('/')[-3]
    if not os.path.isdir(ensemble_save_folder):
        os.mkdir(ensemble_save_folder)
        print 'folder was created :  ',ensemble_save_folder #e.g ./ensemble/cropped_original_fundus_300x300/
    sum_predict={}
    for i,subfolder_name in enumerate(subfolder_names):
        target_model_folder=os.path.join( args.model_dir  , subfolder_name+'/') ## target_model_folder = ./cnn_model/fundus/1/
        result_save_folder = os.path.join(ensemble_save_folder , subfolder_name+'/') #./ensemble/cropped_original_fundus_300x300/1/
        if not os.path.isdir(result_save_folder):
            os.mkdir(result_save_folder ) #target_model_folder = ./ensemble/cropped_original_fundus_300x300/1/
            print 'folder was created!' , result_save_folder
        target_dict = fundus_eval.eval_from_numpy_image(path_dir= args.path_dir , model_dir=target_model_folder)
        f=open(os.path.join(result_save_folder  , 'result.pkl') , 'w')
        pickle.dump(target_dict , f)
        if i==0:
            sum_predict=target_dict
        else:
            for key in sum_predict.keys:
                sum_predict[key] += target_dict[key]

    for key in sum_predict.keys:
        sum_predict[key]=sum_predict[key]/len(subfolder_names) #len(subfolder_names) => the number of model paths

    print 'the number of model' ,len(subfolder_files)

