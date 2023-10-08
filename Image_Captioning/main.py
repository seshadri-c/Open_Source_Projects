import random
import os
from dataloader import *
from model import *
import torch.optim as optim

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
device = torch.device("cuda" if use_cuda else "cpu")

def train_val_test_split(img_folder, annotation_path):

    lines = open(annotation_path, "r").readlines()
    lines = lines[1:]

    #Populated a dictionary in the format of key=img and value = [cap1, cap2, ...]
    image_caption_dict = {}
    for l in lines:
        img = l.split('\t')[0].split('.jpg')[0]+".jpg"
        cap = l.split('\t')[1].strip()
        if img in image_caption_dict.keys():
            image_caption_dict[img].append(cap)
        else:
            image_caption_dict[img] = [cap]

    images = list(image_caption_dict.keys())
    random.seed(10)
    random.shuffle(images)

    #Splitting the list of images into 70, 10, 20 % split in train, val, test respectively.
    train_images = images[:int(0.7*len(images))]
    val_images = images[int(0.7*len(images)):int(0.8*len(images))]
    test_images = images[int(0.8*len(images)):]

    train_list = [(os.path.join(img_folder, img), cap) for img in train_images for cap in image_caption_dict[img] if os.path.exists(os.path.join(img_folder, img))]
    val_list = [(os.path.join(img_folder, img), cap) for img in val_images for cap in image_caption_dict[img] if os.path.exists(os.path.join(img_folder, img))]
    test_list = [(os.path.join(img_folder, img), cap) for img in test_images for cap in image_caption_dict[img] if os.path.exists(os.path.join(img_folder, img))]

    return train_list, val_list, test_list


def train_epoch(train_loader, model, criterion, optimizer):

    model = model.to(device)
    model.train()
    for img_tensor, cap_tensor in train_loader:
        img_tensor = img_tensor.to(device)
        cap_tensor = cap_tensor.to(device)
        outputs = model(img_tensor, cap_tensor[:-1])
        train_loss = criterion(outputs.reshape(-1, outputs.shape[2]), cap_tensor.reshape(-1))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        


def val_epoch():
    pass
def test_epoch():
    pass

def train_val_test(train_loader, val_loader, test_loader, model, loss, optimizer):
    train_epoch(train_loader, model, loss, optimizer)

def main():

    data_folder = "/ssd_scratch/cvit/seshadri_c/Flickr-30K/Flickr-30K"
    image_folder = os.path.join(data_folder, "flickr30k_images")
    annotation_path = os.path.join(data_folder, "results_20130124.token")
    train_list, val_list, test_list = train_val_test_split(image_folder, annotation_path)

    #Preparing Dictionary
    words = []
    for _, cap in train_list:
        cap = re.sub('[^a-zA-Z ]+', '', cap.lower()).split()
        words.extend(cap)
    words = list(set(words))
    words.sort()
    words = ["<BOS>", "<EOS>", "<PAD>", "<UNK>"] + words
    word2int = {}
    int2word = {}
    for id, word in enumerate(words):
        word2int[word] = id
        int2word[id] = word
    batch_size = 32
    train_loader = load_data(train_list, word2int, batch_size=batch_size, num_workers=10)
    val_loader = load_data(val_list, word2int, batch_size=batch_size, num_workers=10)
    test_loader = load_data(test_list, word2int, batch_size=batch_size, num_workers=10)
    model = CNNtoRNN(embed_size=512, hidden_size=256, vocab_size=len(word2int), num_layers=3).to(device)
    loss = nn.CrossEntropyLoss(ignore_index=word2int["<PAD>"])
    learning_rate = 3e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_val_test(train_loader, val_loader, test_loader, model, loss, optimizer)
main()
