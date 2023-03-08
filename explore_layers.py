import numpy as np
from torchvision import models
import torch
from torch import nn
#from yolov5.utils.augmentations import letterbox
from utils.augmentations import letterbox

basepath = '/media/hdda/danosa/congr21/ma/torchpoints/'
modelZoo = {
  'resnet18':         ((224, 224), 'layer4',   f'{basepath}resnet18-f37072fd.pth'),
  'resnet34':         ((224, 224), 'layer4',   f'{basepath}resnet34-b627a593.pth'),
  'resnet50':         ((224, 224), 'layer4',   f'{basepath}resnet50-0676ba61.pth'),
  'resnet101':        ((224, 224), 'layer4',   f'{basepath}resnet101-63fe2227.pth'),
  'resnet152':        ((224, 224), 'layer4',   f'{basepath}resnet152-394f9c45.pth'),
  'wide_resnet50_2':  ((224, 224), 'layer4',   f'{basepath}wide_resnet50-0676ba61.pth'),
  'wide_resnet101_2': ((224, 224), 'layer4',   f'{basepath}wide_resnet101_2-32ee1156.pth'),
  'vgg11_bn':         ((224, 224), 'features', f'{basepath}vgg11_bn-6002323d.pth'),
  'vgg13_bn':         ((224, 224), 'features', f'{basepath}vgg13_bn-abd245e5.pth'),
  'vgg16_bn':         ((224, 224), 'features', f'{basepath}vgg16_bn-6c64b313.pth'),
  'vgg19_bn':         ((224, 224), 'features', f'{basepath}vgg19_bn-c79401a0.pth'),
}

class FeatureExtractor(nn.Module):
    def __init__(self, input_size, model, device, output_layer = None):
        super().__init__()
        self.input_size = input_size
        reversed_layers = tuple(model._modules.keys())[::-1]
        for l in reversed_layers:
            if l != output_layer:
                model._modules.pop(l)
            else:
                break
        self.net = nn.Sequential(model._modules).to(device)
        #self.similarity = nn.CosineSimilarity(dim=0)
        self.eval()

    def forward(self,imgs,device):
      with torch.no_grad():
        imga = np.empty((len(imgs), 3, self.input_size[0], self.input_size[1]), np.float32)
        for i, img in enumerate(imgs):
          #letterbox, HWC to CHW, BGR to RGB
          imga[i] = letterbox(img, new_shape=self.input_size, auto=False, scaleFill=False, scaleup=True)[0].transpose((2, 0, 1))[::-1]
          # normalize to 0..1
          imga[i] /= 255
          # normalize as intended for all torch pretrained models: transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          imga[i,0] -= 0.485
          imga[i,1] -= 0.456
          imga[i,2] -= 0.406
          imga[i,0] /= 0.229
          imga[i,1] /= 0.224
          imga[i,2] /= 0.225
        imga = torch.from_numpy(imga).to(device)
        fet = self.net(imga)
        fet = torch.reshape(fet, (len(imgs), -1,))
        return fet

def getFeatureExtractor(modelName, device, input_size=None, output_layer=None, weights=None):
  if input_size is None or output_layer is None or weights is None:
    input_size, output_layer, weights = modelZoo[modelName]
  model = eval(f'models.{modelName}(pretrained=False)')
  model.load_state_dict(torch.load(weights))
  fe = FeatureExtractor(input_size, model, device, output_layer)
  return fe

def showLayers(model, showBrief=True, showDetailed=True):
  if showBrief:
    print('  BRIEF:')
    for i, l in enumerate(model._modules.keys()):
      #import code; code.interact(local=vars())
      print(f'    Layer {i:03d}: {l}')
  if showDetailed:
    print('  DETAILED:')
    for i, l in enumerate(model._modules.keys()):
      #import code; code.interact(local=vars())
      print(f'    Layer {i:03d}: {l}. Spec: {model._modules[l]}')

def showModels():
  #specs = ('resnet18', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'wide_resnet50_2', 'wide_resnet101_2')
  specs = (
    #'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
      "vgg11",    "vgg11_bn",    "vgg13",    "vgg13_bn",    "vgg16",    "vgg16_bn",    "vgg19",    "vgg19_bn",
  )
  for i, s in enumerate(specs):
    print(f'NETWORK: {s}')
    model = eval(f'models.{s}(pretrained=False)')
    #model.load_state_dict(torch.load("/media/hdda/danosa/congr21/ma/torchpoints/resnet18-f37072fd  .pth"))
    showLayers(model)
    print('')


