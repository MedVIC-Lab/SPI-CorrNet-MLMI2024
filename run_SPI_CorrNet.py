import os
import math
import argparse
import json
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from utils.misc import *
from trainers.mesh2ssm3d import *
from utils.dataset import *
from torch.optim.lr_scheduler import LambdaLR
import trimesh
import numpy as np
import pandas as pd
import itertools
from utils.misc import *

def make_directory(sdir):
	if not os.path.exists(sdir):
		os.makedirs(sdir)


def write_files(particle_dir, names, particle_list):
	output_file = []
	for n,p in zip(names, particle_list):
		n = n.split(".")[0].split("/")[-1] + ".particles"
		# print(n)
		try:
			p = p.detach().cpu().numpy()
		except:
			p = p 
		try:
			np.savetxt(particle_dir + n, np.reshape(p,(-1,3)))
		except:
			np.savetxt(particle_dir + n, np.reshape(p,(-1,1)))
		
		output_file.append(particle_dir + n)
	return output_file


def calculate_point_to_mesh_distance(m,p):
	mesh = trimesh.load(m)
	points = np.loadtxt(p)
	c = trimesh.proximity.ProximityQuery(mesh)
	p2mDist = c.signed_distance(points)
	return p2mDist

torch.cuda.empty_cache() 

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="configs/la.json")
parser.add_argument('--machine', type=str, default='sci')
parser.add_argument('--image_data', type=str, default='orthogonal')
parser.add_argument('--z_spacing', type=int, default=1)
parser.add_argument('--train_size', type=float, default=1)

args = parser.parse_args()
machine = args.machine
image_data = args.image_data
z_spacing = args.z_spacing
train_size = args.train_size

with open(args.config, 'rt') as f:
	t_args = argparse.Namespace()
	t_args.__dict__.update(json.load(f))
	args = parser.parse_args(namespace=t_args)



print(args)
seed_all(args.seed)

args.image_data = image_data
args.z_spacing = z_spacing
args.train_size = train_size


if args.logging:
	log_dir = get_new_log_dir(args.log_root, prefix='SPI-CorrNet', postfix='_' + args.tag if args.tag is not None else '')
	args.log_dir = log_dir
	logger = get_logger('train', log_dir)
	writer = torch.utils.tensorboard.SummaryWriter(log_dir)
	ckpt_mgr = CheckpointManager(log_dir)
	log_hyperparams(writer, args)
else:
	logger = get_logger('train', None)
	writer = BlackHole()
	ckpt_mgr = BlackHole()
logger.info(args)

if(args.image_data == 'orthogonal'):
	train_dset = CommonDataset(args, partition='train')
	val_dset = CommonDataset(args, partition ='val')

elif(args.image_data == 'full'):
	train_dset = CommonDatasetFullImages(args, partition='train')
	val_dset = CommonDatasetFullImages(args, partition ='val')
	

train_iter = DataLoader(
	train_dset,
	batch_size=args.train_batch_size,
	num_workers=8,
	shuffle= True
)

val_iter = DataLoader(
	val_dset,
	batch_size=args.val_batch_size,
	num_workers=8,
	drop_last=True,
	shuffle = True
)
if(args.image_data == 'full'):
	args.img_dims = train_iter.dataset.input_images[0].shape[1:]

template = np.loadtxt(args.data_dir + "/"+args.template_type)
template = torch.from_numpy(template).type(torch.float)

args.input_x_T = template
args.scheduler = None
if(args.deterministic_encoder==True):
	args.num_samples = 1


# Model
logger.info('Building model...')
model = Mesh2SSM3D(args)
model.set_template(args.input_x_T)
logger.info(repr(model))


# Define your model optimizer choices
optimizer_choices = {
	'adam': torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4),
	'adamw': torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4),
	'sgd': torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4),
}



def get_mean_training_correspondences(epoch):
	model.eval()
	dataset_size = len(train_dset)
	mean_correspondence = None
	for data in train_iter:
		vertices = data['pointcloud'].to(args.device)
		idx = data['idx'].to(args.device)		
		with torch.no_grad():
			
			correspondences, _ = model.predict(vertices=vertices, idx=idx, num_samples=None)
			if(mean_correspondence == None):
				mean_correspondence = correspondences.sum(axis=0)
			else:
				mean_correspondence = mean_correspondence + correspondences.sum(axis=0)
	mean_correspondence = mean_correspondence/dataset_size
	np_array = (mean_correspondence.reshape((-1,3))).detach().cpu().numpy()
	np.savetxt( f'{log_dir}/mean_correspondence_{epoch}.particles', np_array)
	return mean_correspondence.reshape((-1,3)).detach().cpu()


def eval(test_argsname, image_inference=True):
	with open(test_argsname, 'rt') as f:
		t_args = argparse.Namespace()
		t_args.__dict__.update(json.load(f))
		test_args = parser.parse_args(namespace=t_args)
	template = np.loadtxt(test_args.data_dir + "/" + test_args.template_type)
	template = torch.from_numpy(template).type(torch.float)
	

	test_args.input_x_T = template
	save_dir = test_args.log_dir + "/results/"
	make_directory(save_dir)
	test_logger = get_logger('test', save_dir)
	for k, v in vars(test_args).items():
		test_logger.info('[ARGS::%s] %s' % (k, repr(v)))

	model = Mesh2SSM3D(test_args)
	test_logger.info('Loading model...')
	test_args.ckpt_path = test_args.log_dir + "/ckpt_best_.pt"
	checkpoint = torch.load(test_args.ckpt_path, map_location = test_args.device)
	model.load_state_dict(checkpoint['state_dict'])
	model.set_template(test_args.input_x_T)
	
	for partition in ['train', 'test', 'val']:
		if(test_args.image_data == 'orthogonal'):
			test_dset = CommonDataset(test_args, partition = partition)
		else:
			test_dset = CommonDatasetFullImages(test_args, partition = partition)

		test_iter = DataLoader(test_dset,
		batch_size=1,
		num_workers=8,
		drop_last=True
		)
		correspondences_pred_dir = save_dir + partition + "_correspondences_pred/"
		uncertainty_dir = save_dir + partition + "_uncertainty/"
		make_directory(uncertainty_dir)
		make_directory(correspondences_pred_dir)
				
		names = []
		corr_p2mDist = []
		corr_particle_list = []
		corr_chamfer_dist_l1 = []
		corr_chamfer_dist_l2 = []
		ale_uncs = []
		model.eval()

		for data in test_iter:
			vertices = data['pointcloud'].to(test_args.device)
			label = data['true_pointcloud'].to(test_args.device)
			n = data['name']
			val_batch_size = vertices.shape[0]
			
			with torch.no_grad():
				if(image_inference == True):
					y_mean, y_log_var = model.predict(images=data['image'], num_samples = test_args.num_samples)
					correspondences, _ = model.predict(images=data['image'])
				else:
					y_mean, y_log_var = model.predict(vertices=vertices, idx = data['idx'].to(args.device))
					correspondences = y_mean
			
			ale_unc = np.exp(y_log_var.detach().cpu().numpy())
			# save per-particle uncertinity for each sample
			name = n[0].split(".")[0].split("/")[-1] + '.npy'
			np.savetxt(uncertainty_dir + '/' + name , ale_unc.reshape((-1,3)))

			ale_uncs.append(np.sum(ale_unc))

			# Predicted correspondence Chamfer distance 
			cd_l1,_ = pytorch3d.loss.chamfer_distance(label.reshape((1,-1,3)), correspondences.reshape((1,-1,3)),point_reduction='mean', batch_reduction="mean", norm=1)
			cd_l2,_ = pytorch3d.loss.chamfer_distance(label.reshape((1,-1,3)), correspondences.reshape((1,-1,3)),point_reduction='mean', batch_reduction="mean", norm=2)
			
			corr_chamfer_dist_l1.append(cd_l1.detach().cpu().numpy())
			corr_chamfer_dist_l2.append(cd_l2.detach().cpu().numpy())
			
			corr_particle = write_files(correspondences_pred_dir, n, correspondences)
			
			corr_particle_list.append(corr_particle)

			#point to surface distance 
			for m,p in zip(n, corr_particle):
				m = test_args.data_dir + partition + "/meshes/" + m
				names.append(m)
				p2m = calculate_point_to_mesh_distance(m,p)
				corr_p2mDist.append(p2m)
					
		corr_chamfer_dist_l1 = np.array(corr_chamfer_dist_l1).flatten()
		corr_chamfer_dist_l2 = np.array(corr_chamfer_dist_l2).flatten()
		ale_uncs = np.array(ale_uncs).flatten()

		corr_p2mDist = np.array(corr_p2mDist)
		corr_p2mDist_mean = np.mean(corr_p2mDist,axis=1)
		# find the worst, median, best sample based on point to mesh distance 
		worst_index, best_index = np.argmax(corr_p2mDist_mean), np.argmin(corr_p2mDist_mean)
		median_index = np.argsort(corr_p2mDist_mean)[len(corr_p2mDist_mean)//2]
		labels = ['worst', 'median', 'best']
		indices = [worst_index, median_index, best_index]
		
		corr_particle_list = list(itertools.chain.from_iterable(corr_particle_list))
		files_dict = {'worst':[corr_particle_list[worst_index],worst_index], 'median':[corr_particle_list[median_index],median_index], 'best':[corr_particle_list[best_index],best_index]}
		pd.DataFrame.from_dict(files_dict).to_csv(f'{save_dir}/{partition}_p2m_file_list.csv', index= False)

		# save file names for shapeworks project creation
		project_dict = {'meshes':names, 'corr_particles':corr_particle_list}	
		print(f'meshes: {len(names)}, corr_particles : {len(corr_particle_list)} ')	
		pd.DataFrame.from_dict(project_dict).to_csv(save_dir + partition + "_file_lists.csv",index=False)

		# save all numpy files
		np.save(f'{save_dir}/{partition}_cd_l1.npy', corr_chamfer_dist_l1)
		np.save(f'{save_dir}/{partition}_cd_l2.npy', corr_chamfer_dist_l2)
		np.save(f'{save_dir}/{partition}_p2mdist.npy', corr_p2mDist)
		np.save(f'{save_dir}/{partition}_ale_uncs.npy', ale_uncs)

		# print all metrics
		test_logger.info(f'Partition: {partition}')
		test_logger.info('Correspondence Chamfer distance L1: %.6f  +/- %.6f ' % (np.mean(corr_chamfer_dist_l1), np.std(corr_chamfer_dist_l1)))
		test_logger.info('Correspondence Chamfer distance L2: %.6f  +/- %.6f ' % (np.mean(corr_chamfer_dist_l2), np.std(corr_chamfer_dist_l2)))
		test_logger.info('Correspondence point to mesh distance: %.6f  +/- %.6f ' % (np.mean(corr_p2mDist), np.std(corr_p2mDist)))
		test_logger.info('Correspondence aleotoric uncertainty: %.6f  +/- %.6f ' % (np.mean(ale_unc), np.std(ale_uncs)))


def test(epoch, image_inference = True):
	
	corr_chamfer_dist_l1 = []
	corr_chamfer_dist_l2 = []

	for data in val_iter:
		noisy_vertices = data['pointcloud']
		images = data['image']
		n = data['name']
		idx = data['idx']
		with torch.no_grad():
			if (image_inference):
				correspondences_pred,_ = model.predict(images = images)
			else:
				correspondences_pred,_ = model.predict(vertices = noisy_vertices.to(args.device), idx = idx.to(args.device))
	
		if(args.noise_level>0):
			label = data['true_pointcloud'].to(args.device)
		else:
			label = noisy_vertices.to(args.device)
		

		# Predicted correspondence Chamfer distance 
		val_batch_size = label.shape[0]
		cd_l1,_ = pytorch3d.loss.chamfer_distance(label.reshape((args.val_batch_size,-1,3)), correspondences_pred.reshape((args.val_batch_size,-1,3)),point_reduction='mean', batch_reduction='mean', norm=1)
		cd_l2,_ = pytorch3d.loss.chamfer_distance(label.reshape((args.val_batch_size,-1,3)), correspondences_pred.reshape((args.val_batch_size,-1,3)),point_reduction='mean', batch_reduction='mean', norm=2)
		
		

		corr_chamfer_dist_l1.append(cd_l1.detach().item())
		corr_chamfer_dist_l2.append(cd_l2.detach().item())
	corr_chamfer_dist_l1 = np.array(corr_chamfer_dist_l1)
	corr_chamfer_dist_l2 = np.array(corr_chamfer_dist_l2)

	return np.mean(corr_chamfer_dist_l1), np.mean(corr_chamfer_dist_l2)




batch_iter = int(train_iter.__len__())

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#------------------------- Teacher branch training ------------------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
for param in model.dgcnn.parameters():
	param.requires_grad = True
for param in model.imnet.parameters():
	param.requires_grad = True
for param in model.encoder.parameters():
	param.requires_grad = False


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
logger.info('Start training...')
best_val_cd = float('inf')
best_template = None
try:
	early_stopping_counter = 0
	it = 0
	for epoch in range(args.epochs_ae):
		
		model.train()
		for batch in train_iter:
			vertices = batch['pointcloud'].to(args.device)
			faces = batch['faces'].to(args.device)
			idx = batch['idx'].to(args.device)
			label = batch['true_pointcloud'].to(args.device)
			# Reset grad and model state
			optimizer.zero_grad()
			
			loss, loss_cd, loss_dgcnn = model.get_loss_mesh(vertices, label, faces, idx)
			
			# Backward and optimize
			loss.backward()    
			optimizer.step()
			
			if (it % batch_iter == 0):
				logger.info('[Train] Epoch %04d | Iter %04d | Loss %.6f | Loss CD %.4f |  Loss DGCNN %.4f' \
					% (epoch, it, loss.mean().item(), loss_cd.mean().item(), loss_dgcnn.mean().item()))
			it = it +1

			#write outputs
			writer.add_scalar('train/loss_cd', loss_cd.mean().item(), it)
			writer.add_scalar('train/loss_dgcnn', loss_dgcnn.mean().item(), it)
			writer.add_scalar('train/loss', loss, it)
			writer.flush()

		if epoch%args.template_update_frequency == 0:
			mean_correspondences = get_mean_training_correspondences(epoch)
			if(epoch!=0):
				model.set_template(mean_correspondences)

		# validation loop to plot predicted correspondences and sampled template
		if epoch % args.model_save_freq == 0 or epoch == args.epochs_ae:
			
			if(args.scheduler!=None):

				opt_states = {
					'optimizer': optimizer.state_dict(),
					'scheduler': scheduler.state_dict(),
					'args': args,
					'current_epoch': epoch
				}
			else:
				opt_states = {
				'optimizer': optimizer.state_dict(),
				'args': args,               
				'current_epoch': epoch
			}

			# save with the name latest.pt so that you don't waste memory saving all the intermediate models
			ckpt_mgr.save(model, args, 0, others=opt_states, step="latest")
			if(epoch == args.epochs_ae):
				filename = log_dir + "/template_" + str(epoch) + ".particles"
				np.savetxt(filename, model.input_x_T)


		# valiadation for getting the best model and early stopping check
		if epoch % args.val_freq == 0 or epoch == args.epochs_ae:
			val_loss_cd_l1, val_loss_cd_l2 = test(epoch, image_inference = False)
			# Early stopping
			logger.info('[Validation] Epoch %04d | Loss CD L1 %.4f | Loss CD L2 %.4f ' \
					% (epoch, val_loss_cd_l1, val_loss_cd_l2))
			if (args.chamfer_dist == 'L1'):
				val_loss = val_loss_cd_l1
			else:
				val_loss = val_loss_cd_l2

			# check for the best model
			if val_loss < best_val_cd:
				best_val_cd = val_loss
				early_stopping_counter = 0
				if(args.scheduler!=None):

					opt_states = {
						'optimizer': optimizer.state_dict(),
						'scheduler': scheduler.state_dict(),
						'args': args,                            
					}
				else:
					opt_states = {
					'optimizer': optimizer.state_dict(),
					'args': args,
					}

				ckpt_mgr.save(model, args, 0, others=opt_states, step='best_ae')
				filename = log_dir + "/best_template.particles"
				np.savetxt(filename, model.input_x_T)
			else:
				early_stopping_counter +=1
				if early_stopping_counter >= args.early_stopping_patience:
					logger.info("Early stopping! No improvement in validation loss.")
					# Epoch when Mesh branch stopped training
					args.epochs_ae = epoch
					break
	test_args = args
	test_args.input_x_T = 0
	test_argsname = log_dir + "/test_args.json"
	with open(test_argsname, 'w') as f:
		json.dump(test_args.__dict__, f, indent =2)

except KeyboardInterrupt:
	logger.info('Terminating...')


# load the best teacher model from current run
ckpt_path = log_dir + "/ckpt_best_ae_.pt"
checkpoint = torch.load(ckpt_path,map_location=args.device)
model.load_state_dict(checkpoint['state_dict'])
best_template = np.loadtxt(log_dir + "/best_template.particles")
best_template = torch.from_numpy(best_template).type(torch.float)
model.set_template(best_template)
	
for param in model.dgcnn.parameters():
	param.requires_grad = False
for param in model.imnet.parameters():
	param.requires_grad = False
for param in model.encoder.parameters():
	param.requires_grad = True

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#------------------------Embedding Alignment ------------------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

logger.info('Start T flank training...')
best_val_loss = float('inf')
best_template = None
try:
	early_stopping_counter = 0
	
	for epoch in range(args.epochs_tf):
		# torch.cuda.empty_cache() 
		model.train()
		for batch in train_iter:
			
			vertices = batch['pointcloud'].to(args.device)
			# if the images are orthogonal slices then this will be a dictionary else this will the images
			images = batch['image']
			idx = batch['idx'].to(args.device)
			
			# Reset grad and model state
			optimizer.zero_grad()

			
			label = batch['true_pointcloud'].to(args.device)
			
			
			loss, loss_cd_image, latent_loss = model.get_loss_image(images, vertices, label, idx)
			
			# Backward and optimize
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1)    
			optimizer.step()
			
			if (it % batch_iter == 0):
				logger.info('[Train] Epoch %04d | Iter %04d | Loss %.4f | Loss Latent %.4f | Loss CD Image %.4f ' \
					% (epoch, it, loss.mean().item(), latent_loss.mean().item(), loss_cd_image.mean().item()))
			it = it +1

			#write outputs
			writer.add_scalar('train/loss_latent', loss.mean().item(), it)
			writer.flush()

		# validation loop to plot predicted correspondences and sampled template
		if epoch % args.model_save_freq == 0 or epoch == args.epochs_tf:
			
			if(args.scheduler!=None):

				opt_states = {
					'optimizer': optimizer.state_dict(),
					'scheduler': scheduler.state_dict(),
					'args': args,
					'current_epoch': epoch
				}
			else:
				opt_states = {
				'optimizer': optimizer.state_dict(),
				'args': args,               
				'current_epoch': epoch
			}

			# save with the name latest.pt so that you don't waste memory saving all the intermediate models
			ckpt_mgr.save(model, args, 0, others=opt_states, step="latest")
			if(epoch == args.epochs_tf):
				filename = log_dir + "/template_" + str(epoch) + ".particles"
				np.savetxt(filename, model.input_x_T)

		# valiadation for getting the best model and early stopping check
		if epoch % args.val_freq == 0 or epoch == args.epochs:
			val_loss_cd_l1, val_loss_cd_l2 = test(epoch, image_inference = True)
			# Early stopping
			logger.info('[Validation] Epoch %04d | Loss CD L1 %.4f | Loss CD L2 %.4f ' \
					% (epoch, val_loss_cd_l1, val_loss_cd_l2))
			if (args.chamfer_dist == 'L1'):
				val_loss = val_loss_cd_l1
			else:
				val_loss = val_loss_cd_l2

			# check for the best model
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				early_stopping_counter = 0
				
				if(args.scheduler!=None):

					opt_states = {
						'optimizer': optimizer.state_dict(),
						'scheduler': scheduler.state_dict(),
						'args': args,                            
					}
				else:
					opt_states = {
					'optimizer': optimizer.state_dict(),
					'args': args,
					}

				ckpt_mgr.save(model, args, 0, others=opt_states, step='best_tf')
				filename = log_dir + "/best_template.particles"
				np.savetxt(filename, model.input_x_T)
			else:
				early_stopping_counter +=1
				if early_stopping_counter >= args.early_stopping_patience:
					logger.info("Early stopping! No improvement in validation loss.")
					# Epoch where T-flank stopped training
					args.epochs_tf = epoch
					break

	test_args = args
	test_args.input_x_T = 0
	test_argsname = log_dir + "/test_args.json"
	with open(test_argsname, 'w') as f:
		json.dump(test_args.__dict__, f, indent =2)

except KeyboardInterrupt:
	logger.info('Terminating...')


# load the best model from current run
ckpt_path = log_dir + "/ckpt_best_tf_.pt"
checkpoint = torch.load(ckpt_path,map_location=args.device)
model.load_state_dict(checkpoint['state_dict'])
best_template = np.loadtxt(log_dir + "/best_template.particles")
best_template = torch.from_numpy(best_template).type(torch.float)
model.set_template(best_template)


slowlearn_epochs = 10
args.early_stopping_patience = args.early_stopping_patience + slowlearn_epochs
initial_lr = 1e-8
image_branch_lr = 1e-5
# train the image encoder only
for param in model.dgcnn.parameters():
	param.requires_grad = False
for param in model.imnet.parameters():
	param.requires_grad = False
for param in model.encoder.parameters():
	param.requires_grad = True



optimizer_i = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=image_branch_lr)
# Define the lambda function for the learning rate scheduler
lr_lambda_i = lambda epoch: initial_lr + (image_branch_lr - initial_lr) * epoch / slowlearn_epochs
scheduler_i = LambdaLR(optimizer_i, lr_lambda_i)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#------------------------- Image PDM training ------------------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

logger.info('Start joint training...')
best_val_loss = float('inf')
best_template = None
try:
	early_stopping_counter = 0
	
	for epoch in range(args.epochs):
		# torch.cuda.empty_cache() 
		model.train()
		for batch in train_iter:
			
			vertices = batch['pointcloud'].to(args.device)
			images = batch['image']#.to(args.device)
			faces = batch['faces'].to(args.device)
			idx = batch['idx'].to(args.device)
			
			# Reset grad and model state
			
			optimizer_i.zero_grad()

			
			label = batch['true_pointcloud'].to(args.device)
			
			
			loss, loss_cd, latent_loss = model.get_loss_imagecd(image=images, vertices=vertices, label=label, faces=faces, idx=idx)
			# Backward and optimize
			loss.backward()    
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
			optimizer_i.step()
			scheduler_i.step()

			
			if (it % batch_iter == 0):
				logger.info('[Train] Epoch %04d | Iter %04d | Loss %.6f | Loss CD %.4f | Loss Latent %.4f ' \
					% (epoch, it, loss.mean().item(), loss_cd.mean().item(), latent_loss.mean().item()))
			it = it +1

			#write outputs
			writer.add_scalar('train/loss_cd', loss_cd.mean().item(), it)
			writer.add_scalar('train/loss_latent', latent_loss.mean().item(), it)
			writer.add_scalar('train/loss', loss, it)
			writer.flush()

		# validation loop to plot predicted correspondences and sampled template
		if epoch % args.model_save_freq == 0 or epoch == args.epochs:
			
			opt_states = {
				'optimizer': optimizer_i.state_dict(),
				'args': args,               
				'current_epoch': epoch,
				
			}

			# save with the name latest.pt so that you don't waste memory saving all the intermediate models
			ckpt_mgr.save(model, args, 0, others=opt_states, step="latest")
			if(epoch == args.epochs):
				filename = log_dir + "/template_" + str(epoch) + ".particles"
				np.savetxt(filename, model.input_x_T)


		# valiadation for getting the best model and early stopping check
		if epoch % args.val_freq == 0 or epoch == args.epochs:
			val_loss_cd_l1, val_loss_cd_l2 = test(epoch, image_inference = True)
			# Early stopping
			logger.info('[Validation] Epoch %04d | Loss CD L1 %.4f | Loss CD L2 %.4f ' \
					% (epoch, val_loss_cd_l1, val_loss_cd_l2))
			if (args.chamfer_dist == 'L1'):
				val_loss = val_loss_cd_l1
			else:
				val_loss = val_loss_cd_l2

			# check for the best model
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				early_stopping_counter = 0
				
				if(args.scheduler!=None):

					opt_states = {
						
						'optimizer_i': optimizer_i.state_dict(),
						'scheduler': scheduler.state_dict(),
						'args': args,                            
					}
				else:
					opt_states = {
					
					'optimizer_i': optimizer_i.state_dict(),
					'args': args,
					}

				ckpt_mgr.save(model, args, 0, others=opt_states, step='best')
				filename = log_dir + "/best_template.particles"
				np.savetxt(filename, model.input_x_T)
			else:
				early_stopping_counter +=1

				if early_stopping_counter >= args.early_stopping_patience:
					logger.info("Early stopping! No improvement in validation loss.")
					break
	
	args.input_x_T = 0
	test_argsname = log_dir + "/test_args.json"
	with open(test_argsname, 'w') as f:
		json.dump(args.__dict__, f, indent =2)
		


except KeyboardInterrupt:
	logger.info('Terminating...')


try:
	eval(test_argsname)
except KeyboardInterrupt:
	logger.info('Terminating...')