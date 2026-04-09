from loaders.loaders import KspSensImgLoader
from torch.utils.data import DataLoader

data_path = '' # Provide your actual data path here
device = 'cuda:0'
batch_size = 1
shuffle = False
num_workers = 0

ds = KspSensImgLoader(data_path,device=device)
kspace,coils,img,fname=ds[0]
_,C,M,N = kspace.shape
shape = [1,M,N]
del kspace,coils,img,fname

loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=False)
