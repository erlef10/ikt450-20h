import os
import numpy as np
import torch
import torchvision
import json
import skimage.draw
import cv2
import random

from engine import train_one_epoch, evaluate
import utils
import transforms as T
import torchvision.transforms as TT

from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

CLASS_NAMES = ['BG', 'Balloon']

class BalloonDataset(object):
    def __init__(self, dataset_dir, subset, transforms):
        self.dataset_dir = os.path.join(dataset_dir, subset)
        self.transforms = transforms

        self.imgs = list(sorted([f for f in os.listdir(os.path.join(dataset_dir, subset)) if not f.endswith('.json')]))
        
        # from mask_rcnn example
        self.annotations = json.load(open(os.path.join(dataset_dir, subset, 'via_region_data.json')))
        self.annotations = list(self.annotations.values())
        self.annotations = [a for a in self.annotations if a['regions']]

    def get_masks(self, image, polygons):
        masks = np.zeros([len(polygons), image.height, image.width], dtype=np.uint8)

        for i, p in enumerate(polygons):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            masks[i, rr, cc] = 1

        return torch.as_tensor(masks, dtype=torch.uint8)

    def get_boxes(self, polygons):
        boxes = []
        for points in polygons:
            xmin = np.min(points['all_points_x'])
            xmax = np.max(points['all_points_x'])
            ymin = np.min(points['all_points_y'])
            ymax = np.max(points['all_points_y'])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return boxes, area

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        annotation = [a for a in self.annotations if a['filename'] == self.imgs[idx]][0]

        if type(annotation['regions']) is dict:
            polygons = [r['shape_attributes'] for r in annotation['regions'].values()]
        else:
            polygons = [r['shape_attributes'] for r in annotation['regions']] 

        boxes, area = self.get_boxes(polygons)
        masks = self.get_masks(img, polygons)

        labels = torch.ones(len(self.imgs), dtype=torch.int64)
        image_id = torch.tensor([idx])

        # required by the damn torchvision example
        iscrowd = torch.zeros(len(self.imgs), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["masks"] = masks
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# https://haochen23.github.io/2020/06/fine-tune-mask-rcnn-pytorch.html

def get_coloured_mask(mask):
    colours = [
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [80, 70, 180],
        [250, 80, 190],
        [245, 145, 50],
        [70, 150, 250],
        [50, 190, 190]
    ]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_prediction(model, device, img_path, confidence):
    img = Image.open(img_path)
    transform = TT.Compose([TT.ToTensor()])
    img = transform(img)
    img = img.to(device)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]

    return masks, pred_boxes, pred_class, pred_score[:pred_t+1]

def segment_instance(model, device, img_path, confidence=0.5):
    masks, boxes, pred_cls, confidence = get_prediction(model, device, img_path, confidence)

    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    for i in range(len(masks)):
        rgb_mask = get_coloured_mask(masks[i])
        img = cv2.addWeighted(rgb_mask, 0.4, img, 1, 0, img)

        pred_str = '{}: {}%'.format(pred_cls[i], str(int(confidence[i] * 100)))

        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 0, 255), thickness=2)
        cv2.putText(img, pred_str, (boxes[i][0][0], int(boxes[i][0][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1)

    cv2.imwrite('result.png', img)

def main(inference):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = 2
    num_epochs = 15

    datasets = {
        'train': BalloonDataset('balloon', 'train', get_transform(train=True)),
        'test': BalloonDataset('balloon', 'val', get_transform(train=False))
    }

    data_loaders = {
        'train': torch.utils.data.DataLoader(
            datasets['train'], 
            batch_size=2, 
            shuffle=True,
            collate_fn=utils.collate_fn
        ),
        'test': torch.utils.data.DataLoader(
            datasets['test'], 
            batch_size=1, 
            shuffle=False,
            collate_fn=utils.collate_fn
        )
    }

    model = get_model_instance_segmentation(num_classes).to(device)

    if not inference:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(num_epochs):
            train_one_epoch(model, optimizer, data_loaders['train'], device, epoch, print_freq=10)
            scheduler.step()
            evaluate(model, data_loaders['test'], device=device)

        torch.save(model.state_dict(), 'mask-rcnn-balloon.pt')
    else:
        model.load_state_dict(torch.load('mask-rcnn-balloon.pt'))
        model.eval()

        segment_instance(model, device, './balloon/val/14898532020_ba6199dd22_k.jpg', confidence=0.6)

if __name__ == '__main__':
    main(inference=True)