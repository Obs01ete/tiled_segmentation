import os
import cv2
import time
import numpy as np

import torch
import torch.utils.data

from accuracy import RollingAccuracy
from net import SegmentationNetwork


def worker_init_fn(worker_id):
    # Here both worker_id and PID are important.
    # worker_id helps the number to be different for diffenent workers
    # PID changes at every run of generator (i.e. every epoch) and
    # therefore randomizes for different epochs.
    np.random.seed(np.random.get_state()[1][0] + worker_id + int(os.getpid()))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, feature_img, gt_img, net_reso_hw, fake_size=1024):
        self._feature_img = cv2.imread(feature_img)
        gt_img = cv2.imread(gt_img)
        self._gt_img = (gt_img[:, :, 0] > 128).astype(np.long)
        self._fake_size = fake_size
        self._net_reso_hw = net_reso_hw

    def __len__(self):
        return self._fake_size

    def __getitem__(self, index):
        h, w = self._net_reso_hw
        feature_img_hwc = self._feature_img[0:h, 0:w]
        feature_img_chw = np.transpose(feature_img_hwc, (2, 0, 1))
        feature_img_float = feature_img_chw.astype(np.float32) / 255
        gt_img = self._gt_img[0:h, 0:w]
        feature_tensor = torch.tensor(feature_img_float)
        gt_tensor = torch.tensor(gt_img)
        return feature_tensor, gt_tensor


class Trainer:
    def __init__(self):
        has_gpu = torch.cuda.device_count() > 0
        if has_gpu:
            print(torch.cuda.get_device_name(0))
        else:
            print("GPU not found")
        self.use_gpu = has_gpu

        self._net = SegmentationNetwork()

        paths = [os.path.join("images", p) for p in ("rgb.png", "gt.png")]
        net_reso_hw = (256, 256)
        self._train_dataset = Dataset(*paths, net_reso_hw)

        train_batch_size = 2 # 16
        num_workers = 4

        self._train_loader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )

        self._snapshot_name = "snapshot.pth"

    def train(self):
        num_epochs = 10
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

                if batch_index % 100 == 0:
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

            # Save after every epoch
            torch.save(self._net.state_dict(), self._snapshot_name)

        print("Training done")


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()
