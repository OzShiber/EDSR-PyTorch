import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

def bg_target(queue):
    while True:
        if not queue.empty():
            filename, tensor = queue.get()
            if filename is None: break
            imageio.imwrite(filename, tensor.numpy())

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        
        # --- START: Patched Directory Logic ---
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.test_only:
            # For testing, 'save' specifies the output folder. 'load' is ignored for dir.
            if not args.save:
                args.save = now # Save to a timestamped folder if no name given
            self.dir = os.path.join('..', 'experiment', args.save)
            # In test_only mode, we never load an old PSNR log.
            # self.log remains an empty tensor.
            
        elif args.load:
            # For training, 'load' specifies resuming from an existing experiment
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                try:
                    self.log = torch.load(self.get_path('psnr_log.pt'))
                    print('Continue from epoch {}...'.format(len(self.log)))
                except FileNotFoundError:
                    # This happens if you resume from a checkpoint that never ran eval
                    print(f"NOTE: 'psnr_log.pt' not found in {self.dir}. Creating a new one.")
            else:
                # If dir doesn't exist, start a new session
                args.load = ''
                if not args.save:
                    args.save = now
                self.dir = os.path.join('..', 'experiment', args.save)
                
        else:
            # For starting a new training session
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        # --- END: Patched Directory Logic ---

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        # We don't create model dir in test_only mode
        if not args.test_only:
            os.makedirs(self.get_path('model'), exist_ok=True)
            
        for d in args.data_test:
            # This is the correct results path from the original code
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        # This function might crash if self.log is empty (e.g. test_only)
        if not self.log.numel():
            return
            
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    

    def begin_background(self):
        self.queue = Queue()

        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            # This is the correct results path from the original code
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        # --- FIX for args.betas ---
        kwargs_optimizer['betas'] = (args.beta1, args.beta2)
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # --- FIX for args.decay ---
    # scheduler
    if args.decay_type == 'step':
        scheduler_class = lrs.StepLR
        kwargs_scheduler = {
            'step_size': args.lr_decay,
            'gamma': args.gamma
        }
    elif args.decay_type.startswith('step'):
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler_class = lrs.MultiStepLR
        kwargs_scheduler = {
            'milestones': milestones,
            'gamma': args.gamma
        }
    else:
        # Fallback or error if you have other decay types
        raise ValueError(f'Unknown decay_type: {args.decay_type}')
    # --- END: FIX for args.decay ---

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            # --- START: FIX for optimizer.pt ---
            # This is the new fix. We check if the file exists
            # before trying to load it.
            optimizer_path = self.get_dir(load_dir)
            if os.path.exists(optimizer_path):
                self.load_state_dict(torch.load(optimizer_path))
                if epoch > 1:
                    for _ in range(epoch): self.scheduler.step()
            else:
                # This is safe to ignore in test_only mode
                if not args.test_only:
                     print(f"WARNING: Optimizer file not found at {optimizer_path}. Skipping load.")
            # --- END: FIX for optimizer.pt ---

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            # PyTorch 1.4.0 and newer compatibility
            if hasattr(self.scheduler, 'get_last_lr'):
                return self.scheduler.get_last_lr()[0]
            else:
                return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

