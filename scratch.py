from importlib import reload
from pyroMixturemodel import *

args = Arguments()
adam_args = {
        "lr": args.learning_rate,
}

transform = transforms.Compose([
    transforms.ToTensor(),
    #normalize,
    ])
test_loader = torch.utils.data.DataLoader(
   dataset=datasets.MNIST(
       root='data/',
       train=False,
       download=True,
       transform=transform,
       ),
   batch_size=128,
   shuffle=True,
)
train_loader = torch.utils.data.DataLoader(
   dataset=datasets.MNIST(
       root='data/',
       train=True,
       download=True,
       transform=transform,
       ),
   batch_size=128,
   shuffle=True,
)


test_data = test_loader.dataset.data.float()/255
test_labels = test_loader.dataset.targets
train_data = train_loader.dataset.data.float()/255
train_labels = train_loader.dataset.targets

print("bump")



vae = VAE()
vae.apply(init_weights)


optimizer = Adam(adam_args)
elbo = TraceGraph_ELBO()
svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

vae.cuda()

x,y = iter(train_loader).next()
x = x.cuda()
x


svi.step(x)

train_epoch(svi, train_loader,)

model = train(args, train_loader, test_loader)


x,y = iter(test_loader).next()

plot_images(x)

model.cpu()
x_hat = model.reconstruct_img(x)

recon = x_hat.reshape(-1,1,28,28).detach().cpu()

plot_images(recon)



model2 = VAE(z_dim=20)
model2.apply(init_weights)
svi2 = create_svi(model2,)

train_loop(svi2, model2, train_loader, test_loader)

targets = train_loader.dataset.targets
latent, _ = model2.encoder(train_data)
latent = latent.detach().cpu()


ut.plot_tsne(latent[:5000], targets[:5000], "ooVAE")


reducer = umap.UMAP(random_state=42)
reducer.fit(X=latent)
embedding = reducer.transform(latent)

plt.scatter(embedding[:,0], embedding[:,1], c=targets, cmap='Spectral', s=5)
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset', fontsize=24);




def model():
    pass
