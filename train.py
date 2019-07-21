import os
import cv2
import math
import time
import numpy as np

import torch
import torch.utils.data

from accuracy import RollingAccuracy
from net import SegmentationNetwork


def worker_init_fn(worker_id):
    # Here both worker_id and PID are important.
    # worker_id helps the number to be different for different workers
    # PID changes at every run of generator (i.e. every epoch) and
    # therefore randomizes for different epochs.
    np.random.seed(np.random.get_state()[1][0] + worker_id + int(os.getpid()))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, feature_img_path, gt_img_path, net_reso_hw, fake_size):
        self._feature_img = cv2.imread(feature_img_path)
        gt_img_path = cv2.imread(gt_img_path)
        self._gt_img = (gt_img_path[:, :, 0] > 128).astype(np.long)
        self._fake_size = fake_size
        self._net_reso_hw = net_reso_hw

    def __len__(self):
        return self._fake_size

    def __getitem__(self, index):
        img_h, img_w = self._feature_img.shape[:2]
        net_h, net_w = self._net_reso_hw
        off_x = np.random.randint(0, img_w-net_w)
        off_y = np.random.randint(0, img_h-net_h)
        feature_img_hwc = self._feature_img[off_y:off_y+net_h, off_x:off_x+net_w]
        feature_img_chw = np.transpose(feature_img_hwc, (2, 0, 1))
        feature_img_float = feature_img_chw.astype(np.float32) / 255
        gt_img = self._gt_img[off_y:off_y+net_h, off_x:off_x+net_w]
        feature_tensor = torch.tensor(feature_img_float)
        gt_tensor = torch.tensor(gt_img)
        return feature_tensor, gt_tensor


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, feature_img, net_reso_hw):
        self._feature_img = feature_img
        img_reso_hw = tuple(feature_img.shape[:2])
        self._net_reso_hw = net_reso_hw

        overlap = min(net_reso_hw[0], net_reso_hw[1]) // 4
        self._strides_hw = tuple([d-overlap for d in net_reso_hw])
        self._tile_reso_hw = tuple([int(math.ceil(img_d / stride_d)) \
                                    for img_d, stride_d in zip(img_reso_hw, self._strides_hw)])

        padded_reso_hw = tuple([s*d for s, d in zip(self._strides_hw, self._tile_reso_hw)])
        self._padded_img = np.zeros((*padded_reso_hw, 3), dtype=np.uint8)
        self._padded_img[0:img_reso_hw[0], 0:img_reso_hw[1], :] = feature_img
        pass

    def __len__(self):
        tr = self._tile_reso_hw
        return tr[0]*tr[1]

    def __getitem__(self, index):
        ih = index // self._tile_reso_hw[1]
        iw = index % self._tile_reso_hw[1]
        off_h = ih * self._strides_hw[0]
        off_w = iw * self._strides_hw[1]
        net_h, net_w = self._net_reso_hw
        tile = self._padded_img[off_h:off_h+net_h, off_w:off_w+net_w, :]
        feature_img_chw = np.transpose(tile, (2, 0, 1))
        feature_img_float = feature_img_chw.astype(np.float32) / 255
        feature_tensor = torch.tensor(feature_img_float)
        return feature_tensor

    def compose(self, tile_list):
        big_img = np.zeros(self._padded_img.shape[:2], dtype=np.uint8)
        for index, tile_img in enumerate(tile_list):
            ih = index // self._tile_reso_hw[1]
            iw = index % self._tile_reso_hw[1]
            off_h = ih * self._strides_hw[0]
            off_w = iw * self._strides_hw[1]
            net_h, net_w = self._net_reso_hw
            big_img[off_h:off_h + net_h, off_w:off_w + net_w] = tile_img
        orig_h, orig_w = self._feature_img.shape[:2]
        big_crop = big_img[:orig_h, :orig_w]
        return big_crop


class Trainer:
    def __init__(self, do_train=False):
        has_gpu = torch.cuda.device_count() > 0
        if has_gpu:
            print(torch.cuda.get_device_name(0))
        else:
            print("GPU not found")
        self.use_gpu = has_gpu

        self._net = SegmentationNetwork()

        self._net_reso_hw = (256, 256)

        if do_train:
            paths = [os.path.join("images", p) for p in ("rgb.png", "gt.png")]
            self._train_dataset = Dataset(*paths, self._net_reso_hw, 200)

            train_batch_size = 16
            num_workers = 4

            self._train_loader = torch.utils.data.DataLoader(
                self._train_dataset,
                batch_size=train_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                worker_init_fn=worker_init_fn
            )
        else:
            self._train_dataset = None
            self._train_loader = None

        self._snapshot_name = "snapshot.pth"

        if not do_train:
            load_kwargs = {} if self.use_gpu else {'map_location': 'cpu'}
            self._net.load_state_dict(torch.load(self._snapshot_name, **load_kwargs))

    def train(self):
        num_epochs = 1000
        learning_rate = 0.02

        optimizer = torch.optim.SGD(self._net.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [int(0.9*num_epochs), int(0.95*num_epochs)], gamma=0.1, last_epoch=-1)

        # rolling_accuracy = RollingAccuracy()

        for epoch in range(num_epochs):
            print("Epoch ------ ", epoch)

            self._net.train()

            training_start_time = time.time()

            for batch_index, (feature_tensor, gt_tensor) in enumerate(self._train_loader):
                if self.use_gpu:
                    feature_tensor.cuda()
                    gt_tensor.cuda()

                pred = self._net.forward(feature_tensor)

                loss = self._net.loss(pred, gt_tensor)

                decoded_batch = self._net.decode(pred)

                # rolling_accuracy.add_batch(
                #     gt_tensor.detach().cpu().numpy(),
                #     decoded_batch.detach().cpu().numpy())

                if batch_index % 10 == 0:
                    print("epoch={} batch={} loss={:.4f}".format(
                        epoch, batch_index, loss.detach().cpu().item()
                    ))

                    # ra_dict = rolling_accuracy.get_accuracies()
                    # print("IoU={:.4f}".format(
                    #     ra_dict["iou"],
                    # ))

                    elapsed_minutes = int(time.time() - training_start_time) // 60
                    print("Elapsed {} minutes".format(elapsed_minutes))

                    print()
                    print("-------------------------------")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

            scheduler.step()

            self.test_prediction()

            # Save after every epoch
            torch.save(self._net.state_dict(), self._snapshot_name)

        print("Training done")

    def test_prediction(self):
        feature_image = cv2.imread("images/rgb.png")
        segmentation_image = self.infer(feature_image)
        output_image = (segmentation_image * 255).astype(np.uint8)
        cv2.imwrite("output.png", output_image)

    def infer(self, feature_image):
        inference_dataset = InferenceDataset(feature_image, self._net_reso_hw)

        inference_batch_size = 1
        num_workers = 4

        inference_loader = torch.utils.data.DataLoader(
            inference_dataset,
            batch_size=inference_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        self._net.eval()

        tile_pred_list = []
        with torch.no_grad():
            for batch_index, feature_tensor in enumerate(inference_loader):
                if self.use_gpu:
                    feature_tensor.cuda()

                pred = self._net.forward(feature_tensor)

                decoded_batch = self._net.decode(pred)

                for dec_tensor in decoded_batch:
                    dec_img = dec_tensor.detach().cpu().numpy()
                    tile_pred_list.append(dec_img)

        big_image = inference_dataset.compose(tile_pred_list)
        return big_image


def main():
    trainer = Trainer(do_train=True)
    trainer.train()


if __name__ == "__main__":
    main()
