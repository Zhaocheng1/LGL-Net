import os
import time
import numpy as np
import math
import argparse
from prettytable import PrettyTable
import torch
import random
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model import model_md as create_model
from utils import train_one_epoch, evaluate
from dataloader import get_loader

# image root
test_index_path = '/home/ubuntu/zc/mbit-skin-cancer/dataset/release_v0/meta/test_indexes.csv'
train_index_path = '/home/ubuntu/zc/mbit-skin-cancer/dataset/release_v0/meta/train_indexes.csv'
val_index_path = '/home/ubuntu/zc/mbit-skin-cancer/dataset/release_v0/meta/valid_indexes.csv'
img_info_path = '/home/ubuntu/zc/mbit-skin-cancer/dataset/release_v0/meta/meta.csv'
source_dir = '/home/ubuntu/zc/mbit-skin-cancer/dataset/release_v0/images/'

#label_list
nevus_list = ['blue nevus','clark nevus','combined nevus','congenital nevus','dermal nevus','recurrent nevus','reed or spitz nevus']
basal_cell_carcinoma_list = ['basal cell carcinoma']
melanoma_list = ['melanoma','melanoma (in situ)','melanoma (less than 0.76 mm)','melanoma (0.76 to 1.5 mm)','melanoma (more than 1.5 mm)','melanoma metastasis']
miscellaneous_list = ['dermatofibroma','lentigo','melanosis','miscellaneous','vascular lesion']
SK_list = ['seborrheic keratosis']
label_list = [nevus_list,basal_cell_carcinoma_list,melanoma_list,miscellaneous_list,SK_list]

#seven-point
pigment_network_label_list = [['absent'],['typical'],['atypical']]#'typical:1,atypical:2
streaks_label_list = [['absent'],['regular'],['irregular']]#regular:1, irregular:2
pigmentation_label_list = [['absent'],
                      ['diffuse regular','localized regular'],
                      ['localized irregular','diffuse irregular']]#regular:1, irregular:2
regression_structures_label_list = [['absent'],
                               ['blue areas','combinations','white areas']]# present:1
dots_and_globules_label_list = [['absent'],['regular'],['irregular']]#regular:1, irregular:2
blue_whitish_veil_label_list = [['absent'],['present']]#present:1
vascular_structures_label_list = [['absent'],
                             ['within regression','arborizing','comma','hairpin','wreath'],
                             ['linear irregular','dotted']]

#num of each label
num_label = len(label_list)
list_label = np.array(list(range(num_label)))

num_pigment_network_label = len(pigment_network_label_list)
list_pigment_network_label = np.array(list(range(num_pigment_network_label)))

num_streaks_label = len(streaks_label_list)
list_streaks_label = np.array(list(range(num_streaks_label)))

num_pigmentation_label = len(pigmentation_label_list)
list_pigmentation_label = np.array(list(range(num_pigmentation_label)))

num_regression_structures_label = len(regression_structures_label_list)
list_regression_structures_label = np.array(list(range(num_regression_structures_label)))

num_dots_and_globules_label = len(dots_and_globules_label_list)
list_dots_and_globules_label = np.array(list(range(num_dots_and_globules_label)))

num_blue_whitish_veil_label = len(blue_whitish_veil_label_list)
list_blue_whitish_veil_label = np.array(list(range(num_blue_whitish_veil_label)))

num_vascular_structures_label = len(vascular_structures_label_list)
list_vascular_structures_label = np.array(list(range(num_vascular_structures_label)))

#metadata information
level_of_diagnostic_difficulty_label_list = ['low','medium','high']
evaluation_list = ['flat','palpable','nodular']
location_list = ['back','lower limbs','abdomen','upper limbs','chest',
                  'head neck','acral','buttocks','genital areas']
sex_list = ['female','male']
management_list = ['excision','clinical follow up','no further examination']

num_level_of_diagnostic_difficulty_label_list = len(level_of_diagnostic_difficulty_label_list)
num_evaluation_list = len(evaluation_list)
num_location_list = len(location_list)
num_sex_list = len(sex_list)
num_management_list = len(management_list)

#class_list
class_list = [num_label,
num_pigment_network_label,
num_streaks_label,
num_pigmentation_label,
num_regression_structures_label,
num_dots_and_globules_label,
num_blue_whitish_veil_label,
num_vascular_structures_label]


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def create_optimizer(config, pg):
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(pg, lr=config.lr, betas=(0.9, 0.999))
    elif config.optimizer == 'SGD':
        optimizer = optim.SGD(pg, lr=config.lr, momentum=0.9, nesterov=True)
    elif config.optimizer == 'AdamW':
        optimizer = optim.AdamW(pg, lr=config.lr)
    else:
        raise ValueError("No such optimizer: {}".format(config.optimizer))

    return optimizer

def create_lr_scheduler(optimizer, config):
    if config.lr_scheduler == 'cosineAnnealing':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.0001)
    elif config.lr_scheduler == 'step':
        # For each step_size epoch, the learning rate is multiplied by gamma to perform the attenuation operation
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.epochs, gamma=0.9)
    elif config.lr_scheduler == 'consine':
        # cosine
        lf = lambda x: ((1 + math.cos(x * math.pi / config.epochs)) / 2) * (1 - config.lrf) + config.lrf
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    elif config.lr_scheduler == 'none':
        scheduler = None
    else:
        raise ValueError("No such scheduler: {}".format(config.lr_scheduler))
    return scheduler

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_classes1 = args.num_classes1
        self.num_classes2 = args.num_classes2
        self.out_path = args.out_path
        self.k_fold = args.k_fold
        self.weights_path = args.out_path + "/weights".format(self.k_fold)
        self.pre_weights = args.weights
        self.modalitys = {"M0": ['waiguan'],
                     "M1": ['huijie'],
                     "M2": ['xueliu'],
                     "M3": ['waiguan', 'xueliu'],
                     "M4": ['waiguan', 'huijie'],
                     "M5": ['huijie', 'xueliu'],
                     "M6": ['waiguan', 'huijie', 'xueliu']}
        self.num_moda = "M4"

    def get_loader(self):

        train_loader = get_loader(
            shape=(299, 299), batch_size=self.batch_size, num_workers=0, mode='train')
        val_loader = get_loader(
            shape=(299, 299), batch_size=self.batch_size, num_workers=0, mode='test')

        return train_loader, val_loader

    def get_model(self):
        model = create_model(class_list).to(self.device)

        if self.pre_weights != "":
            if os.path.exists(self.pre_weights):
                weights_dict = torch.load(self.pre_weights, map_location=self.device)
                load_weights_dict = {k: v for k, v in weights_dict.items()
                                     if model.state_dict()[k].numel() == v.numel()}
                print(model.load_state_dict(load_weights_dict, strict=False))
            else:
                raise FileNotFoundError("not found weights file: {}".format(self.pre_weights))
        # 是否冻结权重

        return model

    def get_optimizer_lr_scheduler(self, model):
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = create_optimizer(self.args, pg)
        scheduler = create_lr_scheduler(optimizer, self.args)
        return optimizer, scheduler

    def train(self, model, optimizer, scheduler, train_loader, val_loader, train_num_index):
        print('Start train .....')
        pre_val_avg_acc = 0
        pre_val_diag_acc = 0
        pre_avg_sps_acc = 0
        for epoch in range(self.epochs):
            # train
            train_avg_loss, train_avg_acc, train_diag_acc, train_sps_acc = train_one_epoch(
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                device=self.device,
                batch_size=self.batch_size)

            print(
                "\n[epoch {}] Train AVG Loss: {:.4f}, Train VAG_acc: {:.4f}, Train Diag acc: {:.4f}, Train SPS acc: {:.4f}, learning rate: {}".format(
                    epoch,
                    train_avg_loss,
                    train_avg_acc,
                    train_diag_acc,
                    train_sps_acc,
                    optimizer.param_groups[0]['lr']))

            scheduler.step()
            # validate
            val_avg_acc, val_diag_acc, val_sps_acc = evaluate(model=model, data_loader=val_loader, device=self.device)
            # torch.save(model.state_dict(), "/home/ubuntu/xiaochunlun2/Net_code/mmf_net/weights/model-{}.pth".format(epoch))
            if val_avg_acc >= pre_val_avg_acc:
                pre_val_avg_acc = val_avg_acc
                torch.save(model.state_dict(), "{}/model-best{}.pth".format(self.weights_path, train_num_index))
                print(
                    "the best model is saved {} epoch at {}/model-best{}.pth".format(epoch, self.weights_path, train_num_index))
            if val_diag_acc >= pre_val_diag_acc:
                pre_val_diag_acc = val_diag_acc
                torch.save(model.state_dict(), "{}/model-best-diag{}.pth".format(self.weights_path, train_num_index))
                print(
                    "the best model is saved {} epoch at {}/model-best-diag{}.pth".format(epoch, self.weights_path, train_num_index))
            if val_sps_acc >= pre_avg_sps_acc:
                pre_avg_sps_acc = val_sps_acc
                torch.save(model.state_dict(), "{}/model-best-spc{}.pth".format(self.weights_path, train_num_index))
                print(
                    "the best model is saved {} epoch at {}/model-best-sps{}.pth".format(epoch, self.weights_path, train_num_index))
            print("Train num index {}:The best diag: {}, best spc: {}, best avg: {}".format(train_num_index, pre_val_diag_acc, pre_avg_sps_acc,
                                                                         pre_val_avg_acc))

        return pre_val_diag_acc, pre_avg_sps_acc, pre_val_avg_acc

    def running(self, train_num_index):
        if os.path.exists(self.weights_path) is False:
            os.makedirs(self.weights_path)

        # set_seed(seed=42)

        train_loader, val_loader = self.get_loader()

        model = self.get_model()

        optimizer, scheduler = self.get_optimizer_lr_scheduler(model)

        diag, spc, avg = self.train(model, optimizer, scheduler, train_loader, val_loader, train_num_index)
        return diag, spc, avg

def opt_init_():
    parser = argparse.ArgumentParser()
    ##############################################################################################
    parser.add_argument('--num_classes1', type=int, default=9)  ######二分类这里写2 九分类这里写9########
    parser.add_argument('--num_classes2', type=int, default=2)  ######二分类这里写2 九分类这里写9########
    ##############################################################################################
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0125)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr_scheduler', type=str, default='consine')
    parser.add_argument('--k_fold', type=int, default=10)
    parser.add_argument('--out_path', type=str, default='/home/ubuntu/xiaochunlun2/Diffusion/LGL-Net')
    # 数据集所在根目录
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    train_num = 10
    opt = opt_init_()
    weights_path = opt.out_path + "/weights"
    torch.backends.cudnn.enabled = False
    best_diag = 0
    best_spc = 0
    best_avg = 0
    best_diag_train_id = 0
    best_spc_train_id = 0
    best_avg_train_id = 0
    table = PrettyTable()
    table.field_names = ["", "DIAG_ID", "DIAG_ACC", "SPC_ID", "SPC_ACC", "AVG_ID", "AVG_ACC"]
    for train_num_index in range(train_num):
        trainer = Trainer(opt)
        diag, spc, avg = trainer.running(train_num_index)
        table.add_row(
            ['Train_ID', train_num_index, diag, train_num_index, spc, train_num_index, avg])
        print(table)
        if diag > best_diag:
            best_diag = diag
            best_diag_train_id = train_num_index
        if spc > best_spc:
            best_spc = spc
            best_spc_train_id = train_num_index
        if avg > best_avg:
            best_avg = avg
            best_avg_train_id = train_num_index
    table.add_row(
        ['Best_ID', best_diag_train_id, best_diag, best_spc_train_id, best_spc, best_avg_train_id, best_avg])

    print(table)

    table_path = "{}/train_result.txt".format(weights_path)
    f = open(table_path, "w")
    current_time = time.strftime('%Y_%m_%d %H_%M_%S', time.localtime(time.time()))
    current_weights = "current_weights: " + weights_path
    f.write(current_time + '\n')
    f.write(current_weights + '\n')
    f.write(str(table) + '\n')
    f.close()
