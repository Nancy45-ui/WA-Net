import os
import sys
import json

import torch
import torch.nn as nn

import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

import logging
from set_logger import set_logger1

# import os
# import sys
# sys.path.append("/home/xyx/lsn/newcode_dwt")

# from model_SE_Resnet50_att_avg_dwtlayer4 import SEResNet
# from model_second import SEResNet
from Second_model_resnet import resnet50


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 每次更改
    modelname = "model_second_resnet50_1ceng"
    # dwt_att_se_avg
    modelppp = "1100"
    data_transform = {
        "train_60": transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val_20": transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    data_root = os.path.abspath(os.path.join(
        os.getcwd(), ".."))  # get data root path
    image_path = os.path.join(data_root, "data_set",
                              "lsn_data")  # flower data set path
    assert os.path.exists(
        image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train_60"),
                                         transform=data_transform["train_60"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())

    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices_flower.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 64
    # linux
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # windows
    nw = 1
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val_20"),
                                            transform=data_transform["val_20"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # 不使用迁移学习
    # net = SEResNet(730).to(device)
    net = resnet50(num_classes=730).to(device)
    print(net)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0002)

    epochs = 100
    best_acc = 0.0
    savepaths = '../logs/'+modelppp+"/"
    modepath = savepaths + modelname + ".pth"
    logpath = savepaths + modelname + ".log"
    save_path = modepath
    set_logger1(logpath)
    # save_path = './seresNext_attention73050.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            # loss.backward(retain_graph=True)
            # torch.autograd.set_detect_anomaly(True)
            loss.backward()
            optimizer.step()

            # print statistics1
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        logging.basicConfig(
            format='%(asctime)s %(message)s', level=logging.INFO)
        logging.info('Epoch [{}/{}], Loss: {:.4f},lr:{}'.format(
            epoch + 1,
            epochs,
            loss.item(),
            optimizer.state_dict()['param_groups'][0]['lr'])
        )

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.5f  val_accuracy: %.5f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        logging.basicConfig(
            format='%(asctime)s %(message)s', level=logging.INFO)
        logging.info('[epoch {}] train_loss: {:.5f}  val_accuracy: {:.5f}'.format(
            epoch + 1,
            running_loss / train_steps,
            val_accurate)
        )
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')
    print("best_acc:%.5f " % best_acc)
    print(modepath)


if __name__ == '__main__':
    main()
