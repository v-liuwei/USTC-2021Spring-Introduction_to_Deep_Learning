from model import ResNet
from main import train, validate, plot_history, set_seed, os, pprint
from data import TinyImageNet
from torch.utils.data import DataLoader
from torchvision import transforms


def run(params, plot=False, verbose=False):
    cnn = ResNet(params['block_sizes'], params['res'], params['norm'],
                   params['dropout'][0], params['dropout'][1])
    if verbose:
        print('number of parameters: {}'.format(sum(p.numel() for p in cnn.parameters() if p.requires_grad)))
    cnn, history = train(
        model=cnn,
        train_data=train_dataloader,
        epochs=params['epochs'],
        lr_init=params['lr_init'],
        lr_min=params['lr_min'],
        lr_decay=params['lr_decay'],
        lr_min_delta=params['lr_min_delta'],
        lr_patience=params['lr_patience'],
        val_data=val_dataloader,
        val_min_delta=params['val_min_delta'],
        val_patience=params['val_patience'],
        restore_best_weights=params['restore_best_weights'],
        top=params['top'],
        verbose=verbose,
        device=params['device']
    )
    if plot:
        plot_history(history)
    loss, acc = validate(cnn, val_dataloader, 'val', top=params['top'], verbose=False, device=params['device'])
    return loss, acc


if __name__ == "__main__":
    data_root = '/home/liuwei/projects/DL_exps/exp2/tiny-imagenet-200/'
    batch_size = 256 * 3

    train_dataset = TinyImageNet(data_root, 'train', transforms.Compose([transforms.ToTensor()]))
    val_dataset = TinyImageNet(data_root, 'val', transforms.Compose([transforms.ToTensor()]))
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False)

    default_params = {
        'block_sizes':
            [
                (64, 64, 1),
                (64, 64, 1),
                (64, 64, 1),
                (64, 128, 2),
                (128, 128, 1),
                (128, 128, 1),
                (128, 128, 1),
                (128, 256, 2),
                (256, 256, 1),
                (256, 256, 1),
                (256, 256, 1),
                (256, 256, 1),
                (256, 256, 1),
                (256, 512, 2),
                (512, 512, 1),
                (512, 512, 1),
            ],  # 33 conv layers
        'epochs': 40,
        'res': True,
        'norm': True,
        'dropout': (0., 0.),
        'lr_init': 1e-3,
        'lr_min': 1e-5,
        'lr_decay': 0.5,
        'lr_min_delta': 0.,
        'lr_patience': 2,
        'val_min_delta': 0.,
        'val_patience': 30,
        'top': [1, 5, 10],
        'restore_best_weights': True,
        'device': 'cuda'
    }

    set_seed(17717)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

    logfile = f'logs_test.txt'

    print("default parameters are:")
    pprint(default_params, width=40)
    with open(logfile, 'w') as file:
        file.write("default parameters are:\n")
        pprint(default_params, stream=file, width=40)

    f = open(logfile, 'a')
    loss, acc_list = run(default_params, plot=True, verbose=True)
    info = f'val_loss = {loss}, val_acc = {acc_list}'
    print(info)
    f.write(info + '\n')
    f.close()
    exit()
