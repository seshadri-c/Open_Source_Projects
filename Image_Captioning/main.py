import random
import os
from dataloader import *
from model import *
import torch.optim as optim
from tqdm import tqdm
from einops import rearrange

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
device = torch.device("cuda" if use_cuda else "cpu")


#Function to Save Checkpoint
def save_ckp(checkpoint, checkpoint_path):
    torch.save(checkpoint, checkpoint_path)

#Function to Load Checkpoint
def load_ckp(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

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
    loss = 0.0
    progress_bar = tqdm(enumerate(train_loader))
    for step, (img_tensor, cap_tensor) in progress_bar:
        img_tensor = img_tensor.to(device)
        cap_tensor = cap_tensor.to(device)
        outputs = model(img_tensor, cap_tensor[:-1])
        train_loss = criterion(outputs.reshape(-1, outputs.shape[2]), cap_tensor.reshape(-1))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
        progress_bar.set_description("Iteration : {}/{}\tTraining Loss : {}".format(step+1, len(train_loader), loss/(step+1)))
        progress_bar.refresh()

    return model, optimizer, train_loss

        


def val_epoch():
    pass
def test_epoch(test_loader, model, word2int, int2word):
    
    model = model.to(device)
    model.eval()
    progress_bar = tqdm(enumerate(test_loader))
    max_len=40
    for step, (img_tensor, cap_tensor) in progress_bar:
        img_tensor = img_tensor.to(device)
        cap_tensor = cap_tensor.to(device)

        cap_token  = torch.tensor([word2int["<BOS>"] for i in range(img_tensor.shape[0])]).unsqueeze(0)
        cap_token = cap_token.to(device)
        output_tokens = []
        for i in range(max_len):
            outputs = model(img_tensor, cap_token)
            outputs = torch.argmax(outputs[-1].unsqueeze(0),dim=2)
            cap_token = torch.cat((cap_token,outputs),dim=0)
            output_tokens.append(outputs)
        output_tokens = torch.vstack(output_tokens)
        output_tokens = rearrange(output_tokens, "T B -> B T")
        for sent_tokens in output_tokens:
            words = [int2word[int(word)] for word in sent_tokens]
            sentence = " ".join(words)
            print(sentence)

        exit()


        # features = model.encoderCNN(img_tensor)
        # outputs = model(img_tensor, cap_tensor[:-1])
        # print(img_tensor.shape, cap_tensor.shape, outputs.shape, features.shape)



def train_val_test(train_loader, val_loader, test_loader, model, loss, optimizer, word2int, int2word):

    # checkpoint_folder = "/ssd_scratch/cvit/seshadri_c/model_checkpoint"
    # os.makedirs(checkpoint_folder, exist_ok=True)
    # epoch = 1
    # model, optimizer, train_loss = train_epoch(train_loader, model, loss, optimizer)

    # #Creating Checkpoint
    # checkpoint = {'state_dict': model.state_dict(),'optimizer': optimizer.state_dict(), 'epoch': epoch}
    # checkpoint_path = os.path.join(checkpoint_folder, "Checpoint_{}.pt".format(epoch))

    # #Save Checkpoints
    # save_ckp(checkpoint, checkpoint_path)
    # print("Checkpoint saved succesfully : ", checkpoint_path)

    # #Load Checkpoints
    # model, optimizer, loaded_epoch = load_ckp(checkpoint_path, model, optimizer)
    # print("Checkpoint loaded succesfully from Epoch : ", loaded_epoch)

    test_epoch(test_loader, model, word2int, int2word)

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
    test_loader = load_data(test_list, word2int, batch_size=12, num_workers=10)
    model = CNNtoRNN(embed_size=512, hidden_size=256, vocab_size=len(word2int), num_layers=3).to(device)
    loss = nn.CrossEntropyLoss(ignore_index=word2int["<PAD>"])
    learning_rate = 3e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_val_test(train_loader, val_loader, test_loader, model, loss, optimizer, word2int, int2word)
main()
