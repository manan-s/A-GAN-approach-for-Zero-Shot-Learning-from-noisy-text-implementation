import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias, 0.0)

class _param:
    def __init__(self):
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.rt_dim = rt_dim

# GENERATOR ARCHITECTURE
class generator(nn.Module):
    def __init__(self, text_dim=11083, X_dim=3584):
        super(generator, self).__init__()
        self.rt = nn.Linear(text_dim, rt_dim)
        self.main = nn.Sequential(nn.Linear(z_dim + rt_dim, h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(h_dim, X_dim),
                                  nn.Tanh())

    def forward(self, z, c):
        rt = self.rt(c)
        input = torch.cat([z, rt], 1)
        output = self.main(input)
        return output

#DISCRIMINATOR ARCHITECTURE
class _Discriminator(nn.Module):
    def __init__(self, y_dim=150, X_dim=3584):
        super(_Discriminator, self).__init__()
        # Discriminator net layer one
        self.D_shared = nn.Sequential(nn.Linear(X_dim, h_dim), 
                                      nn.ReLU())
        '''
        self.D_shared = nn.Sequential(nn.Linear(X_dim, 2*h_dim), 
                                      nn.ReLU(),
                                      nn.Linear(2*h_dim, h_dim),
                                      nn.LeakyReLU(0.25))
        '''          
        # discriminating
        self.D_gan = nn.Linear(h_dim, 1)
        #self.D_gan = nn.LeakyReLU(0.1)
        # classifying
        self.D_cls = nn.Linear(h_dim, y_dim)

    def forward(self, input):
        h = self.D_shared(input)
        return self.D_gan(h), self.D_cls(h)

parser = argparse.ArgumentParser()
parser.add_argument('--display_interval', type=int, default=20)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#Hyper params   
args.centroidL = 1
args.learning_rate = 0.0001
args.batchsize = 1000

torch.manual_seed(random.randint(1, 9999))

def train():
    param = _param()
    dataset = LoadDataset(args)
    param.X_dim = dataset.feature_dim

    data_layer = Features(dataset.labels_train, dataset.pfc_feat_data_train, args)

    generator = generator(dataset.text_dim, dataset.feature_dim)
    generator.apply(weights_init)
    
    Discriminator = _Discriminator(dataset.train_cls_num, dataset.feature_dim)
    Discriminator.apply(weights_init)
    #print(Discriminator)

    start_step = 0
    nets = [generator, Discriminator]

    tr_cls_centroid = Variable(torch.from_numpy(dataset.tr_cls_centroid.astype('float32')))
    optimizerD = torch.optim.Adam(Discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

    #training begins
    for it in range(start_step, 3000+1):

        """ Discriminator """
        for _ in range(5):
            data = data_layer.forward()
            input_data = data['data']   
            X = Variable(torch.from_numpy(input_data))

            labels = data['labels'].astype(int) 
            text_feat = np.array([dataset.tt_feature[i,:] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32')))
            y_true = Variable(torch.from_numpy(labels.astype('long')))
            z = Variable(torch.randn(args.batchsize, param.z_dim))

            # REAL
            D_real, C_real = Discriminator(X)
            #print(C_real)
            D_loss_real = torch.mean(D_real)
            #print(D_loss_real)
            #y_true=torch.tensor(y_true, dtype=torch.long, device=device)
            C_loss_real = torch.nn.functional.cross_entropy(C_real, y_true.long())
            DC_loss = 1-D_loss_real + C_loss_real
            #print("DC_LOSS1: ",DC_loss)
            DC_loss.backward()

            # FAKE
            gen_out = generator(z, text_feat).detach()
            fake, falseC = Discriminator(gen_out)
            D_loss_fake = torch.mean(fake)

            #y_true=torch.tensor(y_true, dtype=torch.long, device=device)
            C_loss_fake = torch.nn.functional.cross_entropy(falseC, y_true.long())
            DC_loss = D_loss_fake + C_loss_fake
            #print("DC_LOSS2: ",DC_loss)
            DC_loss.backward()

            optimizerD.step()

            for net in nets:
                net.zero_grad()

        """ Generator """
        for _ in range(1):
            data = data_layer.forward()
            input_data = data['data'] 
            X = Variable(torch.from_numpy(input_data))

            labels = data['labels'].astype(int)  
            text_feat = np.array([dataset.tt_feature[i, :] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32')))
            y_true = Variable(torch.from_numpy(labels.astype('long')))
            z = Variable(torch.randn(args.batchsize, param.z_dim))

            gen_out = generator(z, text_feat)
            fake, falseC = Discriminator(gen_out)
            _,      C_real = Discriminator(X)

            G_loss = torch.mean(fake)

            # Classification loss
            C_loss = (torch.nn.functional.cross_entropy(C_real, y_true.long()) + torch.nn.functional.cross_entropy(falseC, y_true.long()))/2
            GC_loss = C_loss - G_loss
            '''
            # Centroid loss
            Euclidean_loss = Variable(torch.Tensor([0.0]))
            if args.centroidL != 0:
                for i in range(dataset.train_cls_num):
                    sample_idx = (y_true == i).data.nonzero().squeeze()
                    if sample_idx.numel() == 0:
                        Euclidean_loss += 0.0
                    else:
                        G_sample_cls = gen_out[sample_idx, :]
                        Euclidean_loss += (G_sample_cls.mean(dim=0) - tr_cls_centroid[i]).pow(2).sum().sqrt()
                Euclidean_loss *= 1.0/dataset.train_cls_num * args.centroidL
            '''

            total_loss = GC_loss # + Euclidean_loss
            #print("G_LOSS: ",total_loss)
            total_loss.backward()
            optimizerG.step()
            
            for net in nets:
                net.zero_grad()

        if it % args.display_interval == 0:
            acc_real = (np.argmax(C_real.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            acc_fake = (np.argmax(falseC.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            #print(acc_fake)
            print('Iteration-{}...D_LossReal-{}....D_LossFake-{}....G_Loss-{}....Accuracy_real-{}....Accuracy_Fake-{}'.format(it,D_loss_real,D_loss_fake,G_loss,acc_real,acc_fake))
    
train()
