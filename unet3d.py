import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_features, intermediate_features, out_features, p_dropout, batch_norm = True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_features, intermediate_features, 3, padding=1)
        self.conv2 = nn.Conv3d(intermediate_features, out_features, 3,  padding=1)
        
        self.dropout = nn.Dropout(p_dropout)
        
        self.use_bn = batch_norm
        if self.use_bn:
            self.bn1 = nn.BatchNorm3d(intermediate_features)
            self.bn2 = nn.BatchNorm3d(out_features)
            
        
    def forward(self, x, use_dropout=False):
        if use_dropout:
            x = self.dropout(x)
            
        # first convolution 
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x)
        
        if use_dropout:
            x = self.dropout(x)
            
        # second convolution
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        F.relu(x)
        return x

class EncoderBlock(Block):
    def __init__(self, in_features, out_features, p_dropout, batch_norm = True):
        super().__init__(in_features, out_features//2, out_features, p_dropout, batch_norm)
        
class DecoderBlock(Block):
    def __init__(self, in_features, out_features, p_dropout, batch_norm = True):
        super().__init__(in_features, out_features, out_features, p_dropout, batch_norm)    

class Encoder(nn.Module):
    def __init__(self, in_classes, num_blocks, p_dropout, initial_features=64):
        super().__init__()
        
        self.encoder_blocks = torch.nn.ModuleList()
        
        # the first block dimension are: num_classes -> initial_features/2 -> initial_features
        self.encoder_blocks.append(EncoderBlock(in_classes, initial_features, p_dropout))
        
        # for the intermediate blocks the dimensions are: in_dim -> in_dim -> in_dim*2
        dim = initial_features
        for i in range(num_blocks-1):
            self.encoder_blocks.append(EncoderBlock(dim, dim*2, p_dropout))
            dim *= 2
        
        # the final block is treated seperatly since its outputs done need to be saved
        self.final_block = DecoderBlock(dim, dim*2, p_dropout)
        
        self.max = nn.MaxPool3d(2,2)
        
    def forward(self, x, use_dropout=False):
        feature_maps = []
        for block in self.encoder_blocks:
            x = block(x, use_dropout)
            feature_maps.insert(0,x)
            x = self.max(x)
        x = self.final_block(x, use_dropout)
        return x, feature_maps
        
class Decoder(nn.Module):
    def __init__(self,  num_blocks, p_dropout, initial_features=64):
        super().__init__()
        
        self.decoder_blocks = torch.nn.ModuleList()
        self.upconvs = torch.nn.ModuleList()    
        
        # the last block dimension are: inital_features * 2 + inital_features -> inital_features -> initial_features
        # the blocks before that use the same pattern but with doubled feature size
        # the upconv's start with dim inital_features*2 
        dim = initial_features
        for i in range(num_blocks):
            self.decoder_blocks.insert(0,DecoderBlock(dim*3, dim, p_dropout))
            dim *= 2
            self.upconvs.insert(0, nn.ConvTranspose3d(dim, dim, kernel_size=2, stride=2))
        
    def forward(self, x, feature_maps, use_dropout=False):
        for up, block, feature_map in zip(self.upconvs, self.decoder_blocks, feature_maps):
            
            if x.isnan().any():
                print('NAN input')
            x = up(x)
            if x.isnan().any():
                print('NAN after up convolution')
            x = torch.hstack((x, feature_map))
            if x.isnan().any():
                print('NAN after stacking')
            x = block(x, use_dropout)
            if x.isnan().any():
                print('NAN after block')
        return x

class ClassificationHead(nn.Module):
    def __init__(self, out_classes,  inital_features):
        super().__init__()
        self.conv = nn.Conv3d(inital_features, out_classes, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.conv(x))

class Unet3d(nn.Module): 
    def __init__(self, in_classes, out_classes, num_blocks,  initial_features=64, p_dropout=0.3):
        super().__init__()
        self.encoder = Encoder(in_classes, num_blocks, p_dropout, initial_features)
        self.decoder = Decoder(num_blocks, p_dropout, initial_features)
        self.classification = ClassificationHead(out_classes,  initial_features)
        
    def forward(self, x, use_dropout=False):
        if x.isnan().any():
            print('NAN in input')
        x, outputs = self.encoder(x, use_dropout)
        if x.isnan().any():
            print('NAN after encoder')
        for o in outputs:
            if o.isnan().any():
                print('NAN in output features')
        x = self.decoder(x, outputs, use_dropout)
        if x.isnan().any():
            print('NAN after decoder')
        x = self.classification(x)
        if x.isnan().any():
            print('NAN after classification')
        return x
