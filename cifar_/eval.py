import utils
def eval(sess , fetches , feed_dict , test_imgs , test_labs):
    test_imgs_list , test_labs_list =utils.divide_images_labels_from_batch(test_imgs,test_labs , 80)
    print len(test_imgs_list)

