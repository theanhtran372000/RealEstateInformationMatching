import torch.nn as nn
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from tool.predictor import Predictor as Corrector
from tool.utils import extract_phrases


def get_ocr_model(device):
    # Config model
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = False
    config['device'] = device
    
    # Generate model
    predictor = Predictor(config)
    
    return predictor

class VietOCR(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device
        self.predictor = get_ocr_model(device)
        
    def forward(self, x):
        return self.predictor.predict(x)
        
        
class OCRCorrection(nn.Module):
    def __init__(self, weight_path, model_type='seq2seq', device='cuda:0'):
        super().__init__()
        self.weight_path = weight_path
        self.model_type = model_type
        self.device = device
        
        self.corrector = Corrector(device=self.device, model_type=self.model_type, weight_path=self.weight_path)
        
    def forward(self, x):
        txts = x.split(' ')
        if len(txts) <= 1:
            return x
        else:
            return self.corrector.predict(x.strip(), NGRAM=6)