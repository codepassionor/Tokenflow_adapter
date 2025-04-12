import torch
import webdataset as wds
import numpy as np
import cv2
from PIL import Image
import glob
import pickle
import time
from itertools import cycle
from torchvision import transforms

device_ids = list(range(torch.cuda.device_count()))
device_cycle = cycle(device_ids)

'''def preprocess(sample):
    two_frames, text_emb, depth = sample

    device_id = next(device_cycle)
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
    two_frames = pickle.loads(two_frames)
    text_emb = pickle.loads(text_emb)
    depth = pickle.loads(depth)

    datas = []
    for frame in two_frames:
        frame = cv2.resize(frame, (512, 512))
        #print(frame.shape)
        datas.append(frame[None, :, :, :])

    data = np.concatenate(datas, axis=0)
    data = np.transpose(data, (0, 3, 1, 2))

    depth_map = torch.cat((depth[0], depth[1]), dim=0)
    
    #print(data.shape, text_emb.shape)
    out = torch.Tensor(data), torch.Tensor(text_emb), torch.Tensor(depth_map)
    #print(out[0].shape, out[1].shape)
    return out

def get_data_loader(batch_size, num_workers, shard_shuffle=4, sample_shuffle=128):
    # /root/autodl-fs/dataset-msrvtt-with-sd-v2-1/msrvtt-webdataset.shard-*.tar
    # urls = sorted(glob.glob('/root/autodl-fs/dataset-webvid-with-21/webvid-webdataset.shard-*.tar'))
    
    urls = sorted(glob.glob('/root/autodl-tmp/lora_fs/wd_MSRVTT/msrvtt-webdataset.shard-*.tar'))
    # urls = sorted(glob.glob('/root/autodl-fs/dataset-msrvtt-with-sd-v2-1/msrvtt-webdataset.shard-*.tar'))
    
    # urls = urls[:len(urls) // 4]
    # 取 URL 列表的 3%，测试用
    #urls = urls[:int(len(urls) * 0.03)]
    
    dataset = wds.WebDataset(urls, resampled=True).shuffle(shard_shuffle).to_tuple('frames.pyd', 'text_emb.pyd', 'depth_map.pyd').map(preprocess).shuffle(sample_shuffle).batched(batch_size, partial=False).with_epoch(10000) 
    loader = wds.WebLoader(dataset, num_workers=num_workers)
    return loader'''
def preprocess(sample):
    two_frames, text_emb, depth = sample

    device_id = next(device_cycle)
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    '''
    two_frames = torch.load(two_frames, map_location=device)
    text_emb = torch.load(text_emb, map_location=device)
    noise = torch.load(noise, map_location=device)
    latents = torch.load(latents, map_location=device)
    '''
    two_frames = pickle.loads(two_frames)
    text_emb = pickle.loads(text_emb)
    depth = pickle.loads(depth)

    datas = []
    for frame in two_frames:
        frame = cv2.resize(frame, (512, 512))
        #print(frame.shape)
        datas.append(frame[None, :, :, :])

    data = np.concatenate(datas, axis=0)
    data = np.transpose(data, (0, 3, 1, 2))

    depth_map = torch.cat((depth[0], depth[1]), dim=0)
    
    #print(data.shape, text_emb.shape)
    out = torch.Tensor(data), torch.Tensor(text_emb), torch.Tensor(depth_map)
    #print(out[0].shape, out[1].shape)
    return out

def get_data_loader(args, shard_shuffle=4, sample_shuffle=128):
    # 获取数据集的 URL 列表
    urls = sorted(glob.glob("/wd_MSRVTT/" + '/msrvtt-webdataset.shard-*.tar'))
    batch_size = args.train_batch_size
    image_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    # 预处理数据集
    def preprocess_train(examples):
        frames = torch.stack([image_transforms(frame).cuda() for frame in examples[0]])

        text_embs = torch.stack([text_emb.cuda() for text_emb in examples[1]])
        depth_maps = torch.stack([conditioning_image_transforms(depth_map).cuda() for depth_map in examples[2]])

        examples = (frames, text_embs, depth_maps)
        return examples

    # 创建 WebDataset
    dataset = wds.WebDataset(urls, resampled=True).shuffle(shard_shuffle).to_tuple('frames.pyd', 'text_emb.pyd', 'depth_map.pyd').map(preprocess).map(preprocess_train).shuffle(sample_shuffle).batched(batch_size, partial=False).with_epoch(10000)

    '''# 设置数据集的转换
    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        dataset = dataset.with_transform(preprocess_train)'''

    loader = wds.WebLoader(dataset, num_workers=args.dataloader_num_workers)
    return loader


if __name__ == '__main__':
    cnt = 0
    loader = get_data_loader(8, 16)
    t0 = time.time()
    for i in range(2):
        for item in loader:
            data, emb = item
            print(data.shape, emb.shape)
            first = data[0][0]
            v0 = first[0]
            v1 = first[1]
            print(v0.shape, v1.shape)
            cv2.imwrite('{}_{}_1.png'.format(i, cnt), np.transpose(v0.numpy(), (1,2,0))[:, :, ::-1])
            cv2.imwrite('{}_{}_2.png'.format(i, cnt), np.transpose(v1.numpy(), (1,2,0))[:, :, ::-1])
            cnt += 1

        print(cnt)
    t1 = time.time()
    print(t1 - t0)
