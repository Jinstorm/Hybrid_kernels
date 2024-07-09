from CenModel import CentralModel
from DisModel import DistributedModel
from Option_for_cmd import BaseOptions
from utils import *


if __name__ == '__main__':
    model = BaseOptions()
    opt = model.parse()

    X_train, X_test, Y_train, Y_test = data_preprocess(opt)

    if opt.portion == 0:
        CentralModel(opt, X_train, X_test, Y_train, Y_test)
    else:
        DistributedModel(opt, X_train, X_test, Y_train, Y_test)




