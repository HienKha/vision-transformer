import torch
import torch.nn as nn

class DETRModel(nn.Module):
    def __init__(self, num_classes=4, num_birads=4, model='detr_resnet50'):
        super(DETRModel, self).__init__()
        self.num_classes = num_classes
        self.num_birads = num_birads
        
        self.model = torch.hub.load(
            'facebookresearch/detr', 
            model, 
            pretrained=True,
        )
        self.out1 = nn.Linear(
            in_features1=self.model.class_embed.out_features, 
            out_features1=num_classes
        )

        self.out2 = nn.Linear(
            in_features2=self.model.class_embed.out_features, 
            out_features2=num_birads
        )
        
    def forward(self, images):
        d = self.model(images)
        d['pred_logits1'] = self.out1(d['pred_logits'])
        d['pred_logits2'] = self.out2(d['pred_logits'])
        return d
    
    def parameter_groups(self):
        return { 
            'backbone': [p for n,p in self.model.named_parameters()
                              if ('backbone' in n) and p.requires_grad],
            'transformer': [p for n,p in self.model.named_parameters() 
                                 if (('transformer' in n) or ('input_proj' in n)) and p.requires_grad],
            'embed': [p for n,p in self.model.named_parameters()
                                 if (('class_embed' in n) or ('bbox_embed' in n) or ('query_embed' in n)) 
                           and p.requires_grad],
            'final1': self.out1.parameters(),
            'final2': self.out2.parameters()
        }