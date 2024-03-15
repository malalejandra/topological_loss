import torch

from .base_model import BaseModel

from . import networks
from time import time

class Pix2Pix(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        ngf=64,
        ndf=64,
        netG="unet_256",
        netD="basic",
        norm="batch",
        n_layers_D=3,
        dropout=False,
        init_type="normal",
        init_gain=0.02,
        loss_names=["G_GAN", "G_L1", "D_real", "D_fake"],
        visual_names=["real_A", "fake_B", "real_B"],
        lr=0.0002,
        beta1=0.5,
        gan_mode="lsgan",
        direction="AtoB",
        lambda_L1=100,
        device="cuda",
        device_ids=[],
    ):

        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, isTrain=True)
        
        self.netG_name = netG
        #print("netG in model:", self.netG_name)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = loss_names
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = visual_names
        self.device = device
        self.lambda_L1 = lambda_L1
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ["G", "D"]
        else:  # during test time, only load G
            self.model_names = ["G"]
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(
            input_nc=in_channels,
            output_nc=out_channels,
            ngf=ngf,
            netG=netG,
            norm=norm,
            use_dropout=dropout,
            init_type=init_type,
            init_gain=init_gain,
            gpu_ids=device_ids,
        )
        self.direction = direction
        if (
            self.isTrain
        ):  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(
                init_type=init_type,
                init_gain=init_gain,
                input_nc=in_channels + out_channels,
                ndf=ndf,
                netD=netD,
                n_layers_D=n_layers_D,
                norm=norm,
                gpu_ids=device_ids,
            )

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=lr, betas=(beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=lr, betas=(beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, data, target):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.direction == "AtoB"
        self.real_A = data
        self.real_B = target

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

       # if self.netG_name == "dncnn":
            # DnCNN should output noise
        #    self.out_noise = self.netG(self.real_A)  # G(A)
        #    self.fake_B = self.real_A - self.out_noise  # G(A)
        #else:
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self,scalerD):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        with torch.cuda.amp.autocast():
            fake_AB = torch.cat(
                (self.real_A, self.fake_B), 1
            )  # we use conditional GANs; we need to feed both input and output to the discriminator

        
            pred_fake = self.netD(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            # Real
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        
            pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            # combine loss and calculate gradients
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            
        scalerD.scale(self.loss_D).backward()
        
        scalerD.step(self.optimizer_D)
        scalerD.update()

            
        
        #self.loss_D.backward()

    def backward_G(self,scalerG):
        """Calculate GAN and L1 loss for the generator"""
        
        
        with torch.cuda.amp.autocast():
            # First, G(A) should fake the discriminator
        
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            # real_B=target, fake_B=output
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
        scalerG.scale(self.loss_G).backward()
        scalerG.step(self.optimizer_G)
        
        scalerG.update()
        #self.loss_G.backward()

    def optimize_parameters(self,scalerG,scalerD):
        
        self.optimizer_G.zero_grad(set_to_none=True)  # set G's gradients to zero
        self.optimizer_D.zero_grad(set_to_none=True)  # set D's gradients to zero
        
        t = time()
        self.forward()
        forw_time = time() - t  # compute fake images: G(A)

        with torch.cuda.amp.autocast():
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D

        
        self.backward_D(scalerD)  # calculate gradients for D
        #self.optimizer_D.step()  # update D's weights
        
        
        with torch.cuda.amp.autocast():
            # update G
            self.set_requires_grad(
                self.netD, False
            )  # D requires no gradients when optimizing G

        self.backward_G(scalerG)  # calculate graidents for G
        
        
        #self.optimizer_G.step()  # udpate G's weights
        back_time = time() - forw_time
        return forw_time, back_time
