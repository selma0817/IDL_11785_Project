import torch
import torch.utils.data
from tqdm.auto import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
from collections import Counter

from networks import pretrained_convnextv2_tiny
from dataloader import *
import datetime
from datetime import datetime




##data path##
root = "/ix1/hkarim/yip33/kaggle_dataset/imagenet100"
# train_data = AudioDataset(root=root, partition='train-clean-100') #TODO
# val_data =  AudioDataset(root="/ihome/hkarim/yip33/11785/HW3P2/11785-f24-hw3p2", partition = "dev-clean")  # TODO : You can either use the same class with some modifications or make a new one :)
# test_data = AudioDatasetTest(root="/ihome/hkarim/yip33/11785/HW3P2/11785-f24-hw3p2") #TODO

## networks.py
## evaluate
## metrics
## dataloder


import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

run_name = f"yiyan_run{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# get me RAMMM!!!!
import gc
gc.collect()







config = {
    'epochs'        : 5, 
    'batch_size'    : 32,
    'init_lr'       : 2e-3,
    'architecture'  : 'high-cutoff-submission',
    # Add more as you need them - e.g dropout values, weight decay, scheduler parameters
    'dropout'       : 0.2, # changed from 0.1 to 0.2
    'weight_decay'  : 1e-5,
    #'scheduler'     : 'ReduceLROnPlateau',
    'scheduler'     : "CosineAnnealingWarmRestarts",
    'T_0'            : 20,
    'batch_norm'    : True,
    'optimizer'     : 'AdamW',
    'activation'    : 'ReLU',
    "beam_width"    : 5,
    'checkpoint_dir': f"./checkpoints/{run_name}",
}
os.makedirs(config['checkpoint_dir'], exist_ok=True)



#################### load models ####################
model = pretrained_convnextv2_tiny()
model.eval()
model.to(device)
#####################################################




dataset_val, _= build_imagenet_val_dataset(input_size=224)
imagnet_dataloader_val = torch.utils.data.DataLoader(
    dataset=dataset_val,
    batch_size=1,
    num_workers = 0,
)

for inputs, labels in imagnet_dataloader_val:
    print(inputs.shape)
    print(labels.shape)
    break









#################### hyperparams ####################

criterion = torch.nn.CTCLoss() 

optimizer =  torch.optim.AdamW(model.parameters(), lr= config["init_lr"]) # What goes in here?

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience = 3, threshold=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        factor=0.9,
                                                        patience=3,
                                                        threshold=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config["T_0"])
scaler = torch.cuda.amp.GradScaler()
torch.cuda.empty_cache()


#################### hyperparams ####################



#################### send to wandb ####################
import wandb
wandb.login(key="57c916d673703185e1b47000c74bd854db77bcf8")


run = wandb.init(
    name = "asr_1024_embed_size", ## Wandb creates random run names if you skip this field
    reinit = True, ### Allows reinitalizing runs when you re-run this cell
    # run_id = ### Insert specific run id here if you want to resume a previous run
    # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "hw3p2-yiyan", ### Project should be created in your wandb account
    config = config ### Wandb Config for your run
)


#################### train model ####################

from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer):

    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    total_loss = 0

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += loss.item()

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar

        # Another couple things you need for FP16.
        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close() # You need this to close the tqdm bar

    return total_loss / len(train_loader)


#################### validate model ####################


def validate_model(model, val_loader, decoder, phoneme_map= LABELS):

    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    total_loss = 0
    vdist = 0

    for i, data in enumerate(val_loader):

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)

        with torch.inference_mode():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += float(loss)
        vdist += calculate_levenshtein(torch.permute(h, (1, 0, 2)), y, lh, ly, decoder, phoneme_map)

        batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))), dist="{:.04f}".format(float(vdist / (i + 1))))

        batch_bar.update()

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close()
    total_loss = total_loss/len(val_loader)
    val_dist = vdist/len(val_loader)
    return total_loss, val_dist









def validate(model, dataloader):
    correct_top1 = 0
    total_samples = 0
    accumulated_label = 0
    accumulated_predictions = []

    with torch.no_grad():
        for input, label in tqdm(dataloader):

            input, label = input.to('cuda'), label.to('cuda')

            # Forward pass
            output = model(input)

            # Get top-1 predictions
            _, pred_top1 = output.topk(1, dim=1, largest=True, sorted=True)
            
            curr_label = label.item()
            if curr_label==accumulated_label:
                accumulated_predictions.append(pred_top1.item())
            else:
                counter = Counter(accumulated_predictions)
                _, count = counter.most_common(1)[0]
                total_samples += len(accumulated_predictions)
                correct_top1 += count
                accumulated_label = curr_label
                accumulated_predictions = []
    counter = Counter(accumulated_predictions)
    _, count = counter.most_common(1)[0]
    total_samples += len(accumulated_predictions)
    correct_top1 += count
    # Calculate top-1 accuracy percentage
    top1_accuracy = 100 * correct_top1 / total_samples
    return top1_accuracy

top1_accuracy = validate(model, imagnet_dataloader_val)
print(f'Top-1 Accuracy: {top1_accuracy:.2f}%')









def save_model(model, optimizer, scheduler, metric, epoch, path):
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict(),
         metric[0]                  : metric[1],
         'epoch'                    : epoch},
         path
    )





last_epoch_completed = 0
start = last_epoch_completed
end = config["epochs"]
best_lev_dist = float("inf") 



epoch_model_path = os.path.join(config['checkpoint_dir'], 'last.pth')
best_model_path = os.path.join(config['checkpoint_dir'], 'best_cls.pth')

torch.cuda.empty_cache()
gc.collect()

# #TODO: Please complete the training loop

for epoch in range(0, config['epochs']):

    print("\nEpoch: {}/{}".format(epoch+1, config['epochs']))

    curr_lr = float(optimizer.param_groups[0]['lr'])

    train_loss              = train_model(model, train_loader, criterion, optimizer)
    valid_loss, valid_dist  = validate_model(model, val_loader, decoder, phoneme_map= LABELS)
    #scheduler.step(valid_dist)
    scheduler.step()

    print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
    print("\tVal Dist {:.04f}%\t Val Loss {:.04f}".format(valid_dist, valid_loss))


    wandb.log({
        'train_loss': train_loss,
        'valid_dist': valid_dist,
        'valid_loss': valid_loss,
        'lr'        : curr_lr
    })

    save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, epoch_model_path)
    wandb.save(epoch_model_path)
    print("Saved epoch model")

    if valid_dist <= best_lev_dist:
        best_lev_dist = valid_dist
        save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, best_model_path)
        wandb.save(best_model_path)
        print("Saved best model")
run.finish()



