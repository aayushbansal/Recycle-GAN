import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys


class ReCycleGANModel(BaseModel):
    def name(self):
        return 'ReCycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A0 = self.Tensor(nb, opt.input_nc, size, size)
	self.input_A1 = self.Tensor(nb, opt.input_nc, size, size)
	self.input_A2 = self.Tensor(nb, opt.input_nc, size, size)

        self.input_B0 = self.Tensor(nb, opt.output_nc, size, size)
	self.input_B1 = self.Tensor(nb, opt.output_nc, size, size)
	self.input_B2 = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

	self.which_model_netP = opt.which_model_netP
        if opt.which_model_netP == 'prediction':
		self.netP_A = networks.define_G(opt.input_nc, opt.input_nc, 
						opt.npf, opt.which_model_netP, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
		self.netP_B = networks.define_G(opt.output_nc, opt.output_nc, 
						opt.npf, opt.which_model_netP, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
	else:
        	self.netP_A = networks.define_G(2*opt.input_nc, opt.input_nc,
                                        opt.ngf, 'unet_128', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        	self.netP_B = networks.define_G(2*opt.output_nc, opt.output_nc,
                                        opt.ngf, 'unet_128', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
	    self.load_network(self.netP_A, 'P_A', which_epoch)
	    self.load_network(self.netP_B, 'P_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), 
								self.netP_A.parameters(), self.netP_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
	networks.print_network(self.netP_A)
	networks.print_network(self.netP_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A0 = input['A0']
	input_A1 = input['A1']
	input_A2 = input['A2']	

        input_B0 = input['B0']
	input_B1 = input['B1']
	input_B2 = input['B2']	

        self.input_A0.resize_(input_A0.size()).copy_(input_A0)
	self.input_A1.resize_(input_A1.size()).copy_(input_A1)
	self.input_A2.resize_(input_A2.size()).copy_(input_A2)	

        self.input_B0.resize_(input_B0.size()).copy_(input_B0)
	self.input_B1.resize_(input_B1.size()).copy_(input_B1)
	self.input_B2.resize_(input_B2.size()).copy_(input_B2)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A0 = Variable(self.input_A0)
	self.real_A1 = Variable(self.input_A1)
	self.real_A2 = Variable(self.input_A2)

        self.real_B0 = Variable(self.input_B0)
	self.real_B1 = Variable(self.input_B1)
	self.real_B2 = Variable(self.input_B2)

    def test(self):
        real_A0 = Variable(self.input_A0, volatile=True)
	real_A1 = Variable(self.input_A1, volatile=True)
	
        fake_B0 = self.netG_A(real_A0)
	fake_B1 = self.netG_A(real_A1)
	#fake_B2 = self.netP_B(torch.cat((fake_B0, fake_B1),1))
	if self.which_model_netP == 'prediction':
		fake_B2 = self.netP_B(fake_B0, fake_B1)	
	else:
		fake_B2 = self.netP_B(torch.cat((fake_B0, fake_B1),1))

        self.rec_A = self.netG_B(fake_B2).data
        self.fake_B0 = fake_B0.data
	self.fake_B1 = fake_B1.data
	self.fake_B2 = fake_B2.data

        real_B0 = Variable(self.input_B0, volatile=True)
	real_B1 = Variable(self.input_B1, volatile=True)

        fake_A0 = self.netG_B(real_B0)
	fake_A1 = self.netG_B(real_B1)
	#fake_A2 = self.netP_A(torch.cat((fake_A0, fake_A1),1))
        if self.which_model_netP == 'prediction':
		fake_A2 = self.netP_A(fake_A0, fake_A1)	
	else:
		fake_A2 = self.netP_A(torch.cat((fake_A0, fake_A1),1))

        self.rec_B = self.netG_A(fake_A2).data
        self.fake_A0 = fake_A0.data
	self.fake_A1 = fake_A1.data
	self.fake_A2 = fake_A2.data
	
	#pred_A2 = self.netP_A(torch.cat((real_A0, real_A1),1))
	if self.which_model_netP == 'prediction':
		pred_A2 = self.netP_A(real_A0, real_A1)
	else:	
		pred_A2 = self.netP_A(torch.cat((real_A0, real_A1),1))

	self.pred_A2 = pred_A2.data

	#pred_B2 = self.netP_B(torch.cat((real_B0, real_B1),1))
        if self.which_model_netP == 'prediction':
		pred_B2 = self.netP_B(real_B0, real_B1)
	else:
		pred_B2 = self.netP_B(torch.cat((real_B0, real_B1),1))

	self.pred_B2 = pred_B2.data

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B0 = self.fake_B_pool.query(self.fake_B0)
        loss_D_A0 = self.backward_D_basic(self.netD_A, self.real_B0, fake_B0)

        fake_B1 = self.fake_B_pool.query(self.fake_B1)
        loss_D_A1 = self.backward_D_basic(self.netD_A, self.real_B1, fake_B1)

        fake_B2 = self.fake_B_pool.query(self.fake_B2)
        loss_D_A2 = self.backward_D_basic(self.netD_A, self.real_B2, fake_B2)

	pred_B = self.fake_B_pool.query(self.pred_B2)
	loss_D_A3 = self.backward_D_basic(self.netD_A, self.real_B2, pred_B)

        self.loss_D_A = loss_D_A0.data[0] + loss_D_A1.data[0] + loss_D_A2.data[0] + loss_D_A3.data[0]

    def backward_D_B(self):
        fake_A0 = self.fake_A_pool.query(self.fake_A0)
        loss_D_B0 = self.backward_D_basic(self.netD_B, self.real_A0, fake_A0)

        fake_A1 = self.fake_A_pool.query(self.fake_A1)
        loss_D_B1 = self.backward_D_basic(self.netD_B, self.real_A1, fake_A1)

        fake_A2 = self.fake_A_pool.query(self.fake_A2)
        loss_D_B2 = self.backward_D_basic(self.netD_B, self.real_A2, fake_A2)

        pred_A = self.fake_A_pool.query(self.pred_A2)
        loss_D_B3 = self.backward_D_basic(self.netD_B, self.real_A2, pred_A)

        self.loss_D_B = loss_D_B0.data[0] + loss_D_B1.data[0] + loss_D_B2.data[0] + loss_D_B3.data[0]

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_A0 = self.netG_A(self.real_B0)
            idt_A1 = self.netG_A(self.real_B1)
            loss_idt_A = (self.criterionIdt(idt_A0, self.real_B0) + self.criterionIdt(idt_A1, self.real_B1) )* lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            idt_B0 = self.netG_B(self.real_A0)
            idt_B1 = self.netG_B(self.real_A1)
            loss_idt_B = (self.criterionIdt(idt_B0, self.real_A0) + self.criterionIdt(idt_B1, self.real_A1)) * lambda_A * lambda_idt

            self.idt_A = idt_A0.data
            self.idt_B = idt_B0.data
            self.loss_idt_A = loss_idt_A.data[0]
            self.loss_idt_B = loss_idt_B.data[0]

        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        fake_B0 = self.netG_A(self.real_A0)
        pred_fake = self.netD_A(fake_B0)
        loss_G_A0 = self.criterionGAN(pred_fake, True)

        fake_B1 = self.netG_A(self.real_A1)
        pred_fake = self.netD_A(fake_B1)
        loss_G_A1 = self.criterionGAN(pred_fake, True)

	#fake_B2 = self.netP_B(torch.cat((fake_B0,fake_B1),1))
	if self.which_model_netP == 'prediction':
		fake_B2 = self.netP_B(fake_B0,fake_B1)
	else:
		fake_B2 = self.netP_B(torch.cat((fake_B0,fake_B1),1))

	pred_fake = self.netD_A(fake_B2)
	loss_G_A2 = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        fake_A0 = self.netG_B(self.real_B0)
        pred_fake = self.netD_B(fake_A0)
        loss_G_B0 = self.criterionGAN(pred_fake, True)

        fake_A1 = self.netG_B(self.real_B1)
        pred_fake = self.netD_B(fake_A1)
        loss_G_B1 = self.criterionGAN(pred_fake, True)

        #fake_A2 = self.netP_A(torch.cat((fake_A0,fake_A1),1))
	if self.which_model_netP == 'prediction':
		fake_A2 = self.netP_A(fake_A0,fake_A1)
	else:
		fake_A2 = self.netP_A(torch.cat((fake_A0,fake_A1),1))

        pred_fake = self.netD_B(fake_A2)
        loss_G_B2 = self.criterionGAN(pred_fake, True)

	# prediction loss -- 
	#pred_A2 = self.netP_A(torch.cat((self.real_A0, self.real_A1),1))
	if self.which_model_netP == 'prediction':
		pred_A2 = self.netP_A(self.real_A0, self.real_A1)
	else:
		pred_A2 = self.netP_A(torch.cat((self.real_A0, self.real_A1),1))

	loss_pred_A = self.criterionCycle(pred_A2, self.real_A2) * lambda_A
	
	#pred_B2 = self.netP_B(torch.cat((self.real_B0, self.real_B1),1))
	if self.which_model_netP == 'prediction':
		pred_B2 = self.netP_B(self.real_B0, self.real_B1)
	else:
		pred_B2 = self.netP_B(torch.cat((self.real_B0, self.real_B1),1))
	
	loss_pred_B = self.criterionCycle(pred_B2, self.real_B2) * lambda_B

        # Forward recycle loss
        rec_A = self.netG_B(fake_B2)
        loss_recycle_A = self.criterionCycle(rec_A, self.real_A2) * lambda_A

        # Backward recycle loss
        rec_B = self.netG_A(fake_A2)
        loss_recycle_B = self.criterionCycle(rec_B, self.real_B2) * lambda_B

        # Fwd cycle loss 
        rec_A0 = self.netG_B(fake_B0)
        loss_cycle_A0 = self.criterionCycle(rec_A0, self.real_A0) * lambda_A

        rec_A1 = self.netG_B(fake_B1)
        loss_cycle_A1 = self.criterionCycle(rec_A1, self.real_A1) * lambda_A

        rec_B0 = self.netG_A(fake_A0)
        loss_cycle_B0 = self.criterionCycle(rec_B0, self.real_B0) * lambda_B

        rec_B1 = self.netG_A(fake_A1)
        loss_cycle_B1 = self.criterionCycle(rec_B1, self.real_B1) * lambda_B

        # combined loss
        loss_G = loss_G_A0 + loss_G_A1 + loss_G_A2 + loss_G_B0 + loss_G_B1 + loss_G_B2 + loss_recycle_A + loss_recycle_B + loss_pred_A + loss_pred_B + loss_idt_A + loss_idt_B + loss_cycle_A0 + loss_cycle_A1 + loss_cycle_B0 + loss_cycle_B1
        loss_G.backward()

        self.fake_B0 = fake_B0.data
	self.fake_B1 = fake_B1.data
	self.fake_B2 = fake_B2.data
	self.pred_B2 = pred_B2.data

        self.fake_A0 = fake_A0.data
	self.fake_A1 = fake_A1.data
	self.fake_A2 = fake_A2.data
	self.pred_A2 = pred_A2.data
	
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data

        self.loss_G_A = loss_G_A0.data[0] + loss_G_A1.data[0] + loss_G_A2.data[0]
        self.loss_G_B = loss_G_B0.data[0] + loss_G_B1.data[0] + loss_G_B2.data[0]
        self.loss_recycle_A = loss_recycle_A.data[0]
        self.loss_recycle_B = loss_recycle_B.data[0]
	self.loss_pred_A = loss_pred_A.data[0]
	self.loss_pred_B = loss_pred_B.data[0]

        self.loss_cycle_A = loss_cycle_A0.data[0] + loss_cycle_A1.data[0]
        self.loss_cycle_B = loss_cycle_B0.data[0] + loss_cycle_B1.data[0]

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):

        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Recyc_A', self.loss_recycle_A), ('Pred_A', self.loss_pred_A), ('Cyc_A', self.loss_cycle_A), ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Recyc_B',  self.loss_recycle_B), ('Pred_B', self.loss_pred_B), ('Cyc_B', self.loss_cycle_B)])

        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
        return ret_errors

    def get_current_visuals(self):
        real_A0 = util.tensor2im(self.input_A0)
	real_A1 = util.tensor2im(self.input_A1)
	real_A2 = util.tensor2im(self.input_A2)	

        fake_B0 = util.tensor2im(self.fake_B0)
	fake_B1 = util.tensor2im(self.fake_B1)
	fake_B2 = util.tensor2im(self.fake_B2)

        rec_A = util.tensor2im(self.rec_A)

        real_B0 = util.tensor2im(self.input_B0)
	real_B1 = util.tensor2im(self.input_B1)
	real_B2 = util.tensor2im(self.input_B2)

        fake_A0 = util.tensor2im(self.fake_A0)
	fake_A1 = util.tensor2im(self.fake_A1)
	fake_A2 = util.tensor2im(self.fake_A2)

        rec_B = util.tensor2im(self.rec_B)
	
	pred_A2 = util.tensor2im(self.pred_A2)
	pred_B2 = util.tensor2im(self.pred_B2)

        ret_visuals = OrderedDict([('real_A0', real_A0), ('fake_B0', fake_B0), 
				   ('real_A1', real_A1), ('fake_B1', fake_B1),
				   ('fake_B2', fake_B2), ('rec_A', rec_A), ('real_A2', real_A2),
                                   ('real_B0', real_B0), ('fake_A0', fake_A0),
			           ('real_B1', real_B1), ('fake_A1', fake_A1),
				   ('fake_A2', fake_A2), ('rec_B', rec_B), ('real_B2', real_B2),
				   ('real_A2', real_A2), ('pred_A2', pred_A2),
				   ('real_B2', real_B2), ('pred_B2', pred_B2)])
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
	self.save_network(self.netP_A, 'P_A', label, self.gpu_ids)
	self.save_network(self.netP_B, 'P_B', label, self.gpu_ids)
