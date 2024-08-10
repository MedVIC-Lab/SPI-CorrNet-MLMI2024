import os
import math
import argparse
import json
import torch
import torch.utils.tensorboard
from torch.nn import Module
import pytorch3d
import pytorch3d.loss
import sys
sys.path.append("../")
from utils.misc import *

from models.image_branch.encoder import *
from models.mesh_branch.dgcnn import *
from models.imnet import * 

def MSE(predicted, ground_truth):
    return torch.mean((predicted - ground_truth)**2)

class Mesh2SSM3D(Module):
    """docstring for Mesh2SSM3D"""
    def __init__(self, args):
        super(Mesh2SSM3D, self).__init__()
        self.args = args
        if(args.image_data == 'full'):
            self.encoder = ConvolutionalBackbone(args).to(args.device)
        else:
            if(self.args.use_slice=='all'):
                self.encoder = OrthogonalEncoder(args).to(args.device)
            elif(self.args.use_slice in ['axial', 'sagittal', 'coronal']):
                self.encoder = SingleSliceEncoder(args).to(args.device)
            
        self.dgcnn = DGCNN_AE(args).to(args.device)
        self.imnet = ImNet(in_features=args.emb_dims, nf=args.nf,device=args.device,args=args).to(args.device)
        self.imnet.set_template(args,args.input_x_T.numpy())
        self.deterministic_encoder = args.deterministic_encoder    
        self.num_samples = args.num_samples

    def set_template(self,input_x_T):
        self.input_x_T = input_x_T
        self.imnet.set_template(self.args, self.input_x_T.numpy())



    def predict(self, images=None, vertices=None, idx=None, num_samples = None):
        
        if(images != None):
            z = self.encoder(images, num_samples)
            
        else:
            z, _ = self.dgcnn(vertices, idx)
            
        correspondences = self.imnet(z,self.input_x_T.cpu().detach().numpy())

        if(num_samples!=None and num_samples!=1 and vertices==None):
            correspondences = correspondences.reshape((self.num_samples, len(images), -1))
            y_mean = correspondences.mean(0)
            y_log_var = torch.log(correspondences.var(0))
        else:
            y_mean = correspondences
            y_log_var = torch.zeros_like(y_mean)

        return y_mean, y_log_var

    def get_loss_mesh(self, vertices, label, faces=None, idx=None):

        # mesh flank
        batch_size = vertices.shape[0]
        z_mesh, reconstruction = self.dgcnn(vertices,idx)
        
        m_correspondences = self.imnet(z_mesh, self.input_x_T.detach().numpy())

        if(label==None):
            true_x = vertices
        else:
            true_x = label
        

        if(self.args.mse_weight>0):
            loss_dgcnn = F.mse_loss(true_x.reshape((batch_size,-1,3)), reconstruction.reshape((batch_size,-1,3)), reduction='none')
            loss_dgcnn = loss_dgcnn.mean(axis = (2,1))
            loss_dgcnn =  loss_dgcnn.mean()
        else:
            loss_dgcnn = torch.zeros(1).to(vertices.device)

        if self.args.chamfer_dist == 'L1':
            loss_cd_mesh, _ =  pytorch3d.loss.chamfer_distance(true_x.reshape((batch_size,-1,3)), m_correspondences.reshape((batch_size,-1,3)), point_reduction='mean', batch_reduction='mean', norm=1)
        elif self.args.chamfer_dist == 'L2':
            loss_cd_mesh, _ =  pytorch3d.loss.chamfer_distance(true_x.reshape((batch_size,-1,3)), m_correspondences.reshape((batch_size,-1,3)), point_reduction='mean', batch_reduction='mean', norm=2)
        
        

        
        
        loss = loss_cd_mesh + (self.args.mse_weight*loss_dgcnn)
        return loss, loss_cd_mesh, loss_dgcnn

    def get_loss_image(self, image, vertices, label=None, idx= None):
        # Image flank
        batch_size = vertices.shape[0]
        z = self.encoder(image, self.num_samples)
        
        if(self.num_samples>0):
            z_image_mean = z.reshape((self.num_samples, batch_size, -1)).mean(0)
            
        
        with torch.no_grad():
            z_mesh, reconstruction = self.dgcnn(vertices,idx)
        z_image_mean = z_image_mean.reshape(z_mesh.shape)
        
        latent_loss = F.mse_loss(z_image_mean, z_mesh, reduction='none')
        latent_loss = latent_loss.sum(axis = (1))
        latent_loss = latent_loss.mean()

        correspondences = self.imnet(z,self.input_x_T.cpu().detach().numpy())
        if(self.num_samples>0):
            correspondences = correspondences.reshape((self.num_samples, batch_size, -1))
        correspondences_mean = correspondences.mean(0)
        


        if(label==None):
            true_x = vertices
        else:
            true_x = label
        
        if self.args.chamfer_dist == 'L1':
            loss_cd_image, _ =  pytorch3d.loss.chamfer_distance(true_x.reshape((batch_size,-1,3)), correspondences_mean.reshape((batch_size,-1,3)),point_reduction='mean', batch_reduction='mean', norm=1)
            
        elif self.args.chamfer_dist == 'L2':
            loss_cd_image, _ =  pytorch3d.loss.chamfer_distance(true_x.reshape((batch_size,-1,3)), correspondences_mean.reshape((batch_size,-1,3)),point_reduction='mean', batch_reduction='mean', norm=2)

        loss =  latent_loss
        return loss, loss_cd_image, latent_loss
        


    

    def get_loss_imagecd(self, image, vertices, label, faces=None, idx=None):
        # Image flank
        batch_size = vertices.shape[0]
        z = self.encoder(image, self.num_samples)
        
        if(self.num_samples>0):
            z_image_mean = z.reshape((self.num_samples, batch_size, -1)).mean(0)
            

        correspondences = self.imnet(z,self.input_x_T.cpu().detach().numpy())
        if(self.num_samples>0):
            correspondences = correspondences.reshape((self.num_samples, batch_size, -1))
        correspondences_mean = correspondences.mean(0) # y_mean

        with torch.no_grad():
            z_mesh, reconstruction = self.dgcnn(vertices,idx)
        z_image_mean = z_image_mean.reshape(z_mesh.shape)
        latent_loss = F.mse_loss(z_image_mean, z_mesh, reduction='none')
        latent_loss = latent_loss.sum(axis = (1))
        latent_loss = latent_loss.mean()


        if(label==None):
            true_x = vertices
        else:
            true_x = label
        
        
        if self.args.chamfer_dist == 'L1':
            loss_cd, _ =  pytorch3d.loss.chamfer_distance(true_x.reshape((batch_size,-1,3)), correspondences_mean.reshape((batch_size,-1,3)),point_reduction='mean', batch_reduction='mean', norm=1)  
        elif self.args.chamfer_dist == 'L2':
            loss_cd, _ =  pytorch3d.loss.chamfer_distance(true_x.reshape((batch_size,-1,3)), correspondences_mean.reshape((batch_size,-1,3)),point_reduction='mean', batch_reduction='mean', norm=2)
            

        


        loss = latent_loss + loss_cd 

        return loss, loss_cd, latent_loss

    

    def forward(self, images=None, vertices=None, idx=None):
        if(images!=None):
            z = self.encoder(images)
        else:
            z, _ = self.dgcnn(vertices,idx)
            
        correspondences = self.imnet(z,self.input_x_T.cpu().detach().numpy())

        return correspondences