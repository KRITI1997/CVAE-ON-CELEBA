import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import Dataset
from PIL import Image
import torch as nn
from torchvision.utils import save_image

import os
import zipfile 


import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt




class CelebADataset(Dataset):
    def __init__(self, root, attributes_file, transform=None):
        self.root = root
        self.attributes_file=attributes_file
        self.transform = transform

        # Load attribute annotations from text file
        with open(attributes_file) as f:
            lines = f.readlines()
            self.attributes = [line.strip() for line in lines[1:]]
            #print(self.attributes)

        # Load image filenames and attribute labels
        self.image_filenames = []
        self.attribute_labels = []
        with open('/Users/kriti/Desktop/ece792_hw3/list_attr_celeba.txt') as f:
            lines = f.readlines()
            for line in lines[2:]:
                
                filename, *labels = line.strip().split()
                self.image_filenames.append(os.path.join(root, 'img_align_celeba', filename))
                #print(self.image_filenames)
                self.attribute_labels.append([int(label) for label in labels])
                #print(self.attribute_labels)

    def __getitem__(self, index):
        # Load image and convert to tensor
        image = Image.open(self.image_filenames[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load attribute labels and convert to tensor
        attribute_labels = torch.tensor(self.attribute_labels[index], dtype=torch.float32)
        #print(attribute_labels)
        # Resize attribute vector and pass through fully-connected layer
        fc_layer = torch.nn.Linear(40, 64 * 64)
        reshaped_attributes = fc_layer(attribute_labels)
        reshaped_attributes = reshaped_attributes.view(1, 64, 64)

        # Add attribute channel to image tensor
        attribute_channel = torch.zeros_like(image[0]) + reshaped_attributes
        
        image = torch.cat([image, attribute_channel], dim=0)
        #print(image.shape)

        return image, attribute_labels

    def __len__(self):
        return len(self.image_filenames)

# Set up data transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),          #
    #transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
])

# Set up dataset
celeba_dataset = CelebADataset(
    root='/Users/kriti/Desktop/ece792_hw3',
    attributes_file='/Users/kriti/Desktop/ece792_hw3/list_attr_celeba.txt',
    transform=transform
)


# Split dataset into training and testing sets
train_size = int(0.8 * len(celeba_dataset))
test_size =  len(celeba_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(celeba_dataset, [train_size, test_size])

#train_dataset2=torch.utils.data.random_split(test_dataset, [80,000, test_size-80,000])

# Set up data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

iterator=iter(train_loader)
inputs,label=next(iterator)


class Modelcvae(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder=torch.nn.Sequential(
        # encode
        torch.nn.Conv2d(in_channels=4,
                               out_channels=32,
                               kernel_size=3,stride=2,padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.01, inplace=False),
        torch.nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,stride=2,padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.01, inplace=False),

        torch.nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,stride=2,padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.01, inplace=False),

        torch.nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=3,stride=2,padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.01, inplace=False),
        torch.nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=3,stride=2,padding=(1, 1)),
        torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)                    
        )

        self.encode_mu=torch.nn.Linear(512*4,128) 
        #self.encode_mu=torch.nn.Linear(512*4,1)
              #512*64*64
        self.encode_var=torch.nn.Linear(512*4,128)
        #self.encode_mu=torch.nn.Linear(512*4,1)

        self.decode_in=torch.nn.Linear(128+40,512*4) 
        #self.decode_in=torch.nn.Linear(1+40,512*4) 
            #128+40

        self.decoder=torch.nn.Sequential(
             
             torch.nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,padding=1,output_padding=1),
             torch.nn.LeakyReLU(negative_slope=0.01,inplace=False),
             torch.nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,output_padding=1),
             torch.nn.LeakyReLU(negative_slope=0.01,inplace=False),
             torch.nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1),
             torch.nn.LeakyReLU(negative_slope=0.01,inplace=False),
             torch.nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1),
             torch.nn.LeakyReLU(negative_slope=0.01,inplace=False),
             torch.nn.ConvTranspose2d(32,32,kernel_size=3,stride=2,padding=1,output_padding=1))
        
        self.final_layer=torch.nn.Sequential(
            torch.nn.Conv2d(32, out_channels= 3,kernel_size= 3, padding= 1),
            torch.nn.Tanh())
        
    
    def reparametric(self,mu,var):
        standard_dev = torch.exp(0.5 * var)
        epsilon = torch.randn_like(standard_dev)
        return epsilon * standard_dev + mu


    def forward_encode(self,input):
        result= self.encoder(input)
        print("encoder_output",result.shape)
        result = torch.flatten(result, start_dim=1)
        print("flattened encoder output", result.shape)
        mu = self.encode_mu(result)
        log_var = self.encode_var(result)
        #z = self.reparameteric(mu, log_var)
        return mu, log_var

    def final_decode(self,input):
        output=self.decode_in(input)
        
        output= output.view(-1,512,2,2)
        output=self.decoder(output)
        print(output.shape)
        output=self.final_layer(output)
        return output

model=Modelcvae()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_out,input):
    loss=torch.nn.MSELoss()
    output=loss(recon_out,input)
    return output

final_out=[]
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data , labels
        #labels = one_hot(labels, 10)
        mu, log_var= model.forward_encode(data)
        z=model.reparametric(mu, log_var)
        print(z.shape)
        z = torch.cat([z, labels], dim = 1)
        print("shape of z",z.shape)

        recons_output=model.final_decode(z)
        #print(recons_output)
        print("recons output", recons_output.shape)
        print("input data ",data.shape)
        optimizer.zero_grad()
        loss = loss_function(recons_output, data[:,0:3,:,:])
        loss.backward()
        train_loss += loss.detach().numpy()
        optimizer.step()
        #if batch_idx % 10== 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader),
        loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    final_out.append(recons_output)

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data, labels = data, labels
            #labels = one_hot(labels, 10)
            mu, logvar = model.forward_encode(data)
            z=model.reparametric(mu, logvar)
            print(z.shape)
            z = torch.cat([z, labels], dim = 1)
            print("shape of z",z.shape)


            recons_output=model.final_decode(z)
            img=recons_output[0]
            img=img
            print(img.shape)
            save_image(img,'img.png')
            test_loss += loss_function(recons_output, data[:,0:3,:,:]).detach().cpu().numpy()
            

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))




#for epoch in range(1, 8):
     # train(epoch)
      #PATH= "/Users/kriti/Desktop/ece792_hw3/model2_0.8_8.pt"
      #torch.save({
       #    'epoch': epoch,
        #   'model_state_dict': model.state_dict(),
         #  'optimizer_state_dict': optimizer.state_dict(),
            
          #  }, PATH)
#for epoch in range(1,2):
 #      test(epoch)

checkpoint = torch.load('/Users/kriti/Desktop/ece792_hw3/model2_0.8_8.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print('Previously trained model weights state_dict loaded...')
# load trained optimizer state_dict
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print('Previously trained optimizer state_dict loaded...')
epochs = checkpoint['epoch']


#TASK 2 : MANIPULATION OF IMAGES

model.eval()     
with torch.no_grad():
    for i, (images, attributes) in enumerate(test_loader):
        if i == 0:
            att_temp_smiling=torch.from_numpy(np.zeros_like(attributes))
            att_temp_mustache=torch.from_numpy(np.zeros_like(attributes))
            att_temp_sunglasses=torch.from_numpy(np.zeros_like(attributes))
            
            #for j in range(0,128):
             #if att_temp_smiling[j,32] == -1:
            #att_temp_smiling[:,10] = 10
             #if att_temp_mustache[j,23]== -1:
            #att_temp_mustache[:,11]=10
             #if att_temp_sunglasses[j,18]== -1:
            att_temp_sunglasses[:,15]=10
            mean, logvar = model.forward_encode(images)
            z = model.reparametric(mean, logvar)
            print(z.shape)
            z_sunglasses= torch.cat([z, att_temp_sunglasses], dim=1)
            #z_smiling = torch.cat([z, att_temp_smiling], dim=1)
            #z_mustache = torch.cat([z, att_temp_mustache], dim=1)
            #z_sunglasses= torch.cat([z, att_temp_sunglasses], dim=1)
            #recons_output_smi=model.final_decode(z_smiling)
            #recons_output_mus=model.final_decode(z_mustache)
            recons_output_sun=model.final_decode(z_sunglasses)
            save_image(images[:,0:3,:,:],'original.png', nrow=15)
            #save_image(recons_output_smi, 'recon_images_smiling_222.png', nrow=20)
            #save_image(recons_output_mus, 'recon_images_mustache_222.png', nrow=20)
            save_image(recons_output_sun, 'recon_images_sunglasses_222.png', nrow=20)

            
           
# TASK 3 # Morphing two images


iterator=iter(test_loader)
inputs,label=next(iterator)
image_A = inputs[0]
save_image(image_A[0:3,:,:],'IMAGE_a.png')
image_A=torch.unsqueeze(image_A, dim=0)
#save_image(image_A[0:3,:,:],'IMAGE_a.png')
labelsA=label[0]
#print(image_B.shape)
labelsB=label[1]
image_B = inputs[1]
save_image(image_B[0:3,:,:],'IMAGE_b.png')
image_B= torch.unsqueeze(image_B, dim=0)      # load image A from the CelebA dataset
#image_B = inputs[1]
print(image_A.shape)
muA,varA=model.forward_encode(image_A)

muB,varB=model.forward_encode(image_B)

muA = muA.detach().numpy()
muB = muB.detach().numpy()



import numpy as np

num_samples = 10 # number of samples to generate
delta = np.linspace(0, 1, num_samples)

z_interpolated = np.zeros((num_samples, muA.shape[1]))
y_interpolated = labelsA.repeat(num_samples, 1)

for i in range(num_samples):
    z_interpolated[i, :] = (1 - delta[i]) * muA + delta[i] * muB

with torch.no_grad():
    z_interpolated = torch.from_numpy(z_interpolated).float()
    z_interpolated = torch.cat([z_interpolated, y_interpolated], dim = 1)
    print(z_interpolated.shape)
    images_interpolated = model.final_decode(z_interpolated)

    # visualize the interpolated images
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, num_samples, figsize=(20, 4))
for i in range(num_samples):
    axs[i].imshow(images_interpolated[i].permute(1, 2, 0).numpy())
    axs[i].axis('off')
plt.show()






