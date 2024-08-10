import torch
import os
import sys
import glob
import pickle
import numpy as np
from torch.utils.data import Dataset
from .dataset_images import *
from .dataset_meshes import *





class CommonDataset(Dataset):
	"""docstring for CommonDataset"""
	def __init__(self, args, partition):
		super(CommonDataset, self).__init__()
		self.args = args
		self.partition = partition
		
		if(args.z_spacing==1):
			self.image_dir = args.data_dir + f'/{partition}/images/'
		else:
			self.image_dir = args.data_dir + f'/{partition}/images_{args.z_spacing}/'
			
		self.image_extension = args.image_extension
		self.mesh_dir = args.data_dir + f'/{partition}/meshes/'
		self.mesh_extension = args.mesh_extension

		image_files = sorted(glob.glob(f'{self.image_dir}/*.{args.image_extension}'))

		# np.array(all_axial), np.array(all_coronal), np.array(all_sagittal), image_files
		self.axial, self.coronal, self.sagittal , filenames = get_orthogonal_images(self.image_dir, image_files)
		self.axial = torch.FloatTensor(self.axial)
		self.axial = torch.unsqueeze(self.axial, 1)
		self.coronal = torch.FloatTensor(self.coronal)
		self.coronal = torch.unsqueeze(self.coronal, 1)
		self.sagittal = torch.FloatTensor(self.sagittal)
		self.sagittal = torch.unsqueeze(self.sagittal, 1)
		self.names = [os.path.basename(filename).split(".")[0] for filename in filenames]
		self.shuffle_points = args.shuffle_points
		self.data, self.faces_all, self.idx_all, self.max_size, self.scale, self.filename = \
		load_meshes_with_faces(self.mesh_dir, self.partition, self.mesh_extension,self.args.k, self.args.max_gdist)

		if(self.args.tiny_test):
			
			self.data = self.data[:100]
			self.faces_all = self.faces_all[:100]
			# self.idx_all = self.idx_all[:100]
			self.filename = self.filename[:100]
			self.axial = self.axial[:100]
			self.coronal = self.coronal[:100]
			self.sagittal = self.sagittal[:100]
			self.names = self.names[:100]

		self.train_size = args.train_size
		if(self.train_size<1 and partition=='train'):
			total_samples = len(self.data)
			subset_size = int(self.train_size*total_samples)
			index = np.random.choice(total_samples, subset_size, replace = False).tolist()
			self.data = [self.data[i] for i in index]
			self.faces_all = [self.faces_all[i] for i in index]
			self.filename = [self.filename[i] for i in index]
			self.axial = [self.axial[i] for i in index]
			self.coronal = [self.coronal[i] for i in index]
			self.sagittal = [self.sagittal[i] for i in index]
			self.names = [self.names[i] for i in index]

		# self.clip_meshes = args.clip_meshes
		# self.max_size = 5137

	def __getitem__(self, item):
		axial = self.axial[item]
		coronal = self.coronal[item]
		sagittal = self.sagittal[item]

		image_name = self.names[item]
		name = self.filename[item]
		# print(image_name, name)
		assert image_name in name
		# self.max_size = 5137
		pointcloud = self.data[item]
		faces = self.faces_all[item]

		excess = self.max_size - len(pointcloud)
		idx = self.idx_all[name]
		idx_extended = idx
		

		list_idx = list(range(len(pointcloud)))
		if(excess > 0):
			repeat_idx = np.random.randint(0,len(pointcloud),excess)
			list_idx = list_idx + list(repeat_idx)

		pc = pointcloud[list_idx,:]
		idx_extended = idx[list_idx,:]
		faces = faces[list_idx, :]
		
		if(self.shuffle_points):
			s_idx = np.random.randint(0,len(pc),len(pc)).tolist()
			pc = pc[s_idx,:]
			idx_extended = idx_extended[s_idx,:]
			faces = faces[s_idx, :]
		
		pc = torch.from_numpy(pc).type(torch.float)
		
		max_value = pc.max()
		
		if(self.args.noise_level>0):
			noisy_pc = pc + ((self.args.noise_level*max_value) * np.random.randn(*pc.shape))#.to_numpy()
		else:
			noisy_pc = pc

		noisy_pc = noisy_pc.type(torch.float)
		pc = pc.type(torch.float)
		

		data_dict = {
					'pointcloud': noisy_pc,
					'name':name,
					'idx': idx_extended,
					'true_pointcloud': pc,
					'faces': faces,
					'image': {'axial': axial,
							  'coronal': coronal,
					          'sagittal': sagittal}
				}
		
		return data_dict
		
	def __len__(self):

		return len(self.data)





class CommonDatasetFullImages(Dataset):
	"""docstring for CommonDataset"""
	def __init__(self, args, partition):
		super(CommonDatasetFullImages, self).__init__()
		self.args = args
		self.partition = partition
		if(args.z_spacing==1):
			self.image_dir = args.data_dir + f'/{partition}/images/'
		else:
			self.image_dir = args.data_dir + f'/{partition}/images_{args.z_spacing}/'
		self.image_extension = args.image_extension
		self.mesh_dir = args.data_dir + f'/{partition}/meshes/'
		self.mesh_extension = args.mesh_extension

		image_files = sorted(glob.glob(f'{self.image_dir}/*.{args.image_extension}'))

		image_files = sorted(glob.glob(f'{self.image_dir}/*.{args.image_extension}'))
		self.input_images, filenames = get_images(self.image_dir, image_files)
		self.input_images = torch.FloatTensor(self.input_images)
		self.names = [os.path.basename(filename).split(".")[0] for filename in filenames]

		
		self.shuffle_points = args.shuffle_points
		self.data, self.faces_all, self.idx_all, self.max_size, self.scale, self.filename = \
		load_meshes_with_faces(self.mesh_dir, self.partition, self.mesh_extension,self.args.k, self.args.max_gdist)

		if(self.args.tiny_test):
			
			self.data = self.data[:100]
			self.faces_all = self.faces_all[:100]
			# self.idx_all = self.idx_all[:100]
			self.filename = self.filename[:100]
			self.input_images = self.input_images[:100]
			self.names = self.names[:100]

		self.train_size = args.train_size
		if(self.train_size<1 and partition=='train'):
			total_samples = len(self.data)
			subset_size = int(self.train_size*total_samples)
			index = np.random.choice(total_samples, subset_size, replace = False).tolist()
			self.data = [self.data[i] for i in index]
			self.faces_all = [self.faces_all[i] for i in index]
			self.filename = [self.filename[i] for i in index]
			self.input_images = [self.input_images[i] for i in index]
			self.names = [self.names[i] for i in index]

		

	def __getitem__(self, item):
		image = self.input_images[item]

		image_name = self.names[item]
		name = self.filename[item]
		# print(image_name, name)
		# print(image_name, name)
		# assert image_name in name
		# self.max_size = 5137
		pointcloud = self.data[item]
		faces = self.faces_all[item]

		excess = self.max_size - len(pointcloud)
		idx = self.idx_all[name]
		idx_extended = idx
		

		list_idx = list(range(len(pointcloud)))
		if(excess > 0):
			repeat_idx = np.random.randint(0,len(pointcloud),excess)
			list_idx = list_idx + list(repeat_idx)

		pc = pointcloud[list_idx,:]
		idx_extended = idx[list_idx,:]
		faces = faces[list_idx, :]
		
		if(self.shuffle_points):
			s_idx = np.random.randint(0,len(pc),len(pc)).tolist()
			pc = pc[s_idx,:]
			idx_extended = idx_extended[s_idx,:]
			faces = faces[s_idx, :]
		
		pc = torch.from_numpy(pc).type(torch.float)
		
		max_value = pc.max()
		
		if(self.args.noise_level>0):
			noisy_pc = pc + ((self.args.noise_level*max_value) * np.random.randn(*pc.shape))#.to_numpy()
		else:
			noisy_pc = pc

		noisy_pc = noisy_pc.type(torch.float)
		pc = pc.type(torch.float)
		

		data_dict = {
					'pointcloud': noisy_pc,
					'name':name,
					'idx': idx_extended,
					'true_pointcloud': pc,
					'faces': faces,
					'image': image
				}
		
		return data_dict
		
	def __len__(self):

		return len(self.data)