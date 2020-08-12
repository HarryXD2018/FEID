import torch
import torchvision
import collections
import os


def chg_vgg16():
    HOME = os.environ['HOME']
    model_path = "{}/.torch/models/vgg16-397923af.pth".format(HOME)
    if not os.path.exists(model_path):
        model = torchvision.models.vgg16(pretrained=True)
    assert(os.path.exists(model_path))
    
    x = torch.load(model_path)
    val = collections.OrderedDict()
    num_classes = 1
    
    for key in x.keys():
        val[key] = x[key]
    
    val['classifier.6.weight'] = val['classifier.6.weight'][:num_classes, :]
    val['classifier.6.bias'] = val['classifier.6.bias'][:num_classes]
    
    y = {}
    y['state_dict'] = val
    y['epoch'] = 0
    if not os.path.exists('model'):
        os.makedirs('model')
    torch.save(y, 'model/vgg16_{}cls_state_dict.pth'.format(num_classes))

chg_vgg16()
