import sys
sys.path.append("..")
import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from utils import prepare_device

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        #print("!!!optim", self.optimizer)
        self.gan = "pix2pix" in config["arch"]["type"].lower()
        
        
        #if self.gan:
         #   print("###",self.model.optimizer_G,self.model.optimizer_D)
 
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        
        self.device, self.device_ids = prepare_device(config['n_gpu'])
                
        if config.resume is not None:
            
            if not self.gan:
                self._resume_checkpoint(resume_path=config.resume,model=self.model)
            else:
                #print("mnames",self.model.model_names)
                #print("config.resume",config.resume.name, config.resume.parent.parent)
                for n in self.model.model_names:
                    
                        net = getattr(self.model, 'net' + n)
                        if isinstance(net, torch.nn.DataParallel):
                            net = net.module
                        #net = torch.nn.DataParallel(net)
                        print("-------net---------:",type(net), type(self.model))
                        optim = getattr(self.model, 'optimizer_' + n)
                        #print("-------optim---------:",optim)
                        
                        resume_path=config.resume.parent.parent / f"net{n}" / config.resume.name
                        
                        self._resume_checkpoint(resume_path=resume_path, model=net)
                        
                        chkpt = torch.load(resume_path)
                        
                        #print("%%%%",chkpt.keys(), chkpt['optimizer'].keys())
                        optim.load_state_dict(chkpt['optimizer'])
                        
                                                #/f"net{n}")
            
        
            
            
        if len(self.device_ids) > 1 and not self.gan:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
    

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0 and improved :
                if not self.gan:
                    self._save_checkpoint(model=self.model, optimizer=self.optimizer, epoch=epoch, checkpoint_dir=self.checkpoint_dir, save_best=best)
                else:
                    for n in self.model.model_names:
                        net = getattr(self.model, 'net' + n)
                        optim = getattr(self.model, 'optimizer_' + n)
                        
                        self._save_checkpoint(model=net, epoch=epoch, optimizer=optim, checkpoint_dir=self.checkpoint_dir/f"net{n}", save_best=best)
                        
                    

    def _save_checkpoint(self, model, optimizer, epoch, checkpoint_dir, save_best=True):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        arch = type(model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': model.module.state_dict() if len(self.device_ids) > 1 else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        
        filename = str(checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(checkpoint_dir / 'model_best.pth')

            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
            
            
            

    def _resume_checkpoint(self, resume_path, model):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        #print("self_opt",self.optimizer)
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
#        print("chkpt",checkpoint["optimizer"])
        #print("conf",self.config["optimizer"])

        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        pretrained_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
        model.load_state_dict(pretrained_dict)
        if self.optimizer is not None:
            # load optimizer state from checkpoint only when optimizer type is not changed.
            if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
                self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                    "Optimizer parameters not being resumed.")
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        
        
        
    def load_networks(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                
                #load_filename = 
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
