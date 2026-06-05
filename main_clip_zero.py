import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from open_clip import create_model_from_pretrained, get_tokenizer, create_model_and_transforms

LABEL_KEYWORD_LIST_DISEASE = [
    'age-related macular degeneration',
    'diabetic retinopathy',
    # 'pathologic myopia',
    'retinal detachment',
    'retinal vein occlusion',
    # 'uveitis',
    'normal',  # healthy
]

disease_data_dict = {
    'AMD': LABEL_KEYWORD_LIST_DISEASE[0],
    'DR': LABEL_KEYWORD_LIST_DISEASE[1],
    # 'PM': LABEL_KEYWORD_LIST_DISEASE[2],
    'RD': LABEL_KEYWORD_LIST_DISEASE[2],
    'RVO': LABEL_KEYWORD_LIST_DISEASE[3],
    # 'Uveitis': label_list_disease[5],
    'Healthy': LABEL_KEYWORD_LIST_DISEASE[4],  # label_list_disease[6]
}


def main(model_name, pretrained, image_dir, cache_dir, result_dir, args):
    ##############################
    # 1. Create model
    if model_name == 'biomedclip':
        model, _, preprocess = create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir=cache_dir)
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir=cache_dir)
    elif model_name == 'clip':
        if not os.path.isfile(pretrained):
            print('Invalid pretrained model path:', pretrained)
            pretrained = None
        model, preprocess = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384', pretrained=pretrained, cache_dir=cache_dir)
        tokenizer = get_tokenizer('ViT-H-14', cache_dir=cache_dir)
    else:
        raise ValueError('Invalid model name', model_name)

    model = model.to(device=args.device, dtype=args.dtype)
    model.eval()

    ##############################
    # 2. Read images
    image_dict = {os.path.splitext(filename)[0]: os.path.join(dir, filename)
                  for dir, _, filenames in os.walk(image_dir) for filename in filenames
                  if filename.lower().endswith(('.jpg', 'jpeg', '.png', '.tif'))}

    ##############################
    # 3. Define label
    label_list = LABEL_KEYWORD_LIST_DISEASE

    # Expected test.csv format:
    #   Image ID,Diagnosis
    #   000001.png,AMD
    #   000002.png,DR
    #   000003.png,Healthy
    data_dict = {}
    df = pd.read_csv(os.path.join(image_dir, 'test.csv'), usecols=['Image ID', 'Diagnosis'])
    for index, row in list(df.iterrows()):
        image_id = os.path.splitext(row['Image ID'])[0]
        data_dict[image_id] = disease_data_dict[row['Diagnosis']]

    ##############################
    # 4. Extract text features
    print('extract text features ...')
    text = tokenizer(label_list, context_length=model.context_length)
    with torch.no_grad():
        text = text.to(device=args.device)
        text_features = model.encode_text(text)
        text_features = F.normalize(text_features, dim=-1).detach()

    ##############################
    # 5. Main loop
    print('extract image features ...')
    y_pred, y_true, y_prob = [], [], []
    for key, image_path in tqdm(list(image_dict.items())):
        # label processing
        if key not in data_dict:
            continue
        label = data_dict[key]

        image_feature = extract_image_feature(image_path, model, preprocess)

        # Calculate similarity
        with torch.no_grad():
            image_features = torch.from_numpy(image_feature).to(device=args.device, dtype=args.dtype).unsqueeze(0)
            text_probs = torch.sigmoid(image_features @ text_features.T * model.logit_scale.exp())
            text_probs = F.softmax(text_probs, dim=1)[0].detach().cpu().to(dtype=torch.float32).numpy()

        # append
        y_pred.append(np.argmax(text_probs))
        y_true.append(label_list.index(label))
        y_prob.append(text_probs)

    y_prob = np.array(y_prob)

    ##############################
    # 6. Calculate accuracy
    classes = list(range(len(label_list)))
    accuracy_classes = [(sum(1 for p, g in zip(y_pred, y_true) if p == g and g == c) / y_true.count(c)) if y_true.count(c) != 0 else -1 for c in classes]  # with exception handling
    acc = np.mean([a for a in accuracy_classes if a > 0])  # macro average

    y_true_bin = label_binarize(y_true, classes=classes)
    auc_classes = [roc_auc_score(y_true_bin[:, i], y_prob[:, i]) if np.sum(y_true_bin[:, i]) > 0 else -1 for i in classes]
    auc = np.mean([a for a in auc_classes if a > 0])  # macro average

    aupr_classes = [average_precision_score(y_true_bin[:, i], y_prob[:, i]) if np.sum(y_true_bin[:, i]) > 0 else -1 for i in classes]
    aupr = np.mean([a for a in aupr_classes if a > 0])
    print(f' * Acc {acc}, Auc: {auc}, Aupr: {aupr}')

    ##############################
    # 7. Save results
    out_dict = {'label': label_list + ['avg'], 'acc': accuracy_classes + [acc], 'auc': auc_classes + [auc], 'aupr': aupr_classes + [aupr]}
    pd.DataFrame(out_dict).to_csv(os.path.join(result_dir, f'{args.exp_name}.csv'), index=False, encoding='utf-8-sig')


def extract_image_feature(image_path, model, preprocess):
    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        image = image.to(device=args.device, dtype=args.dtype)
        image_features = model.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
    return image_features[0].detach().cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='clip')
    parser.add_argument('--pretrained', default='./models/mmfundusclip/model.pt')
    parser.add_argument('--cache_dir', default='./models/clip')
    parser.add_argument('--result_dir', default='./result')
    parser.add_argument('--image_dir', default='')
    parser.add_argument('--task', default='disease')
    args = parser.parse_args()

    if args.task != 'disease':
        raise ValueError('Invalid task', args.task)

    args.device = torch.device('cuda:0')
    args.dtype = torch.float32  # torch.bfloat16

    # handle defaults
    if args.image_dir == '':
        args.image_dir = './datasets/OpenDataset/'

    # set exp name
    args.exp_name = args.task
    args.exp_name += '_' + os.path.splitext(os.path.basename(args.pretrained))[0]

    # run main
    main(args.model_name, args.pretrained, args.image_dir, args.cache_dir, args.result_dir, args)
